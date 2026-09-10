import { describe, it, expect, beforeAll } from 'react-native-harness';
import {
  loadTensorflowModel,
  type TfliteModel,
  type Tensor,
  type TensorflowModelDelegate,
} from 'react-native-fast-tflite';
import * as jpeg from 'jpeg-js';
import { Platform } from 'react-native';

const MODEL_ASSET = require('../assets/efficientdet.tflite') as number;
const QUANT_MODEL_ASSET = require('../assets/google-quant.tflite') as number;

const CPU_DELEGATES: TensorflowModelDelegate[] = [];
const XNNPACK_DELEGATES: TensorflowModelDelegate[] = ['xnnpack'];
const BENCHMARK_WARMUP_RUNS = 3;
const BENCHMARK_RUNS = 20;

function bytesPerElement(dataType: string): number {
  switch (dataType) {
    case 'uint8':
    case 'int8':
      return 1;
    case 'float16':
    case 'bfloat16':
    case 'int16':
    case 'uint16':
      return 2;
    case 'float32':
    case 'int32':
    case 'uint32':
      return 4;
    case 'float64':
    case 'int64':
    case 'uint64':
      return 8;
    default:
      throw new Error(
        `Unsupported tensor data type for buffer sizing: ${dataType}`,
      );
  }
}

function tensorElementCount(tensor: Tensor): number {
  const dims = tensor.shape;
  if (dims.length === 0) {
    return 1;
  }
  return dims.reduce((a, b) => a * b, 1);
}

function tensorByteLength(tensor: Tensor): number {
  const dims = tensor.shape;
  if (dims.length === 0) {
    return bytesPerElement(tensor.dataType);
  }
  const count = dims.reduce((a, b) => a * b, 1);
  return count * bytesPerElement(tensor.dataType);
}

function zeroedInputBuffer(input: Tensor): ArrayBuffer {
  const len = tensorByteLength(input);
  const buf = new ArrayBuffer(len);
  new Uint8Array(buf).fill(0);
  return buf;
}

function filledInputBuffer(input: Tensor, byte: number): ArrayBuffer {
  const len = tensorByteLength(input);
  const buf = new ArrayBuffer(len);
  new Uint8Array(buf).fill(byte % 256);
  return buf;
}

/** Output buffers are pre-allocated per tensor and reused, so snapshot before re-running. */
function copyOf(buffer: ArrayBuffer): ArrayBuffer {
  return new Uint8Array(new Uint8Array(buffer)).buffer as ArrayBuffer;
}

function buffersEqual(a: ArrayBuffer, b: ArrayBuffer): boolean {
  if (a.byteLength !== b.byteLength) {
    return false;
  }
  const ua = new Uint8Array(a);
  const ub = new Uint8Array(b);
  for (let i = 0; i < ua.length; i++) {
    if (ua[i] !== ub[i]) {
      return false;
    }
  }
  return true;
}

function firstInputTensor(model: TfliteModel): Tensor {
  const tensor = model.inputs[0];
  if (tensor == null) {
    throw new Error('Expected model to declare at least one input tensor');
  }
  return tensor;
}

function tensorSpecsMatch(a: Tensor[], b: Tensor[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    const u = a[i];
    const v = b[i];
    if (u == null || v == null) {
      return false;
    }
    if (
      u.name !== v.name ||
      u.dataType !== v.dataType ||
      u.shape.length !== v.shape.length
    ) {
      return false;
    }
    for (let j = 0; j < u.shape.length; j++) {
      if (u.shape[j] !== v.shape[j]) {
        return false;
      }
    }
  }
  return true;
}

// ImageNet 1001-class index for "giant panda, panda, panda bear"
const GIANT_PANDA_INDEX = 389;

const PANDA_IMAGE_URL =
  'https://images.unsplash.com/photo-1564349683136-77e08dba1ef7?w=400&q=80';

function fetchArrayBuffer(url: string): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'arraybuffer';
    xhr.setRequestHeader('Accept', 'image/jpeg, image/*');
    xhr.onload = () => {
      if (xhr.status === 200) {
        resolve(new Uint8Array(xhr.response as ArrayBuffer));
      } else {
        reject(new Error(`HTTP ${xhr.status}`));
      }
    };
    xhr.onerror = () => reject(new Error('Network error'));
    xhr.send();
  });
}

function resizeAndExtractRGB(
  rgba: Uint8Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): Uint8Array {
  const rgb = new Uint8Array(dstW * dstH * 3);
  const xRatio = srcW / dstW;
  const yRatio = srcH / dstH;
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const srcX = Math.min(Math.floor(x * xRatio), srcW - 1);
      const srcY = Math.min(Math.floor(y * yRatio), srcH - 1);
      const srcIdx = (srcY * srcW + srcX) * 4;
      const dstIdx = (y * dstW + x) * 3;
      rgb[dstIdx] = rgba[srcIdx];
      rgb[dstIdx + 1] = rgba[srcIdx + 1];
      rgb[dstIdx + 2] = rgba[srcIdx + 2];
    }
  }
  return rgb;
}

async function fetchImageAsRGB(
  url: string,
  targetSize: number,
): Promise<Uint8Array> {
  const data = await fetchArrayBuffer(url);
  const decoded = jpeg.decode(data, { useTArray: true });
  return resizeAndExtractRGB(
    decoded.data,
    decoded.width,
    decoded.height,
    targetSize,
    targetSize,
  );
}

function expectAllFloat32Finite(buf: ArrayBuffer, tensor: Tensor): void {
  if (tensor.dataType !== 'float32') {
    return;
  }
  const n = tensorElementCount(tensor);
  const floats = new Float32Array(buf, 0, n);
  for (let i = 0; i < floats.length; i++) {
    expect(Number.isFinite(floats[i])).toBe(true);
  }
}

type InferenceTiming = {
  averageMs: number;
  maxMs: number;
  minMs: number;
};

function nowMs(): number {
  return globalThis.performance.now();
}

function benchmarkRunSync(
  model: TfliteModel,
  input: Tensor,
  warmupRuns: number,
  measuredRuns: number,
): InferenceTiming {
  for (let i = 0; i < warmupRuns; i++) {
    model.runSync([filledInputBuffer(input, i)]);
  }

  let totalMs = 0;
  let minMs = Number.POSITIVE_INFINITY;
  let maxMs = 0;

  for (let i = 0; i < measuredRuns; i++) {
    const inputBuffer = filledInputBuffer(input, i);
    const startMs = nowMs();
    model.runSync([inputBuffer]);
    const elapsedMs = nowMs() - startMs;
    totalMs += elapsedMs;
    minMs = Math.min(minMs, elapsedMs);
    maxMs = Math.max(maxMs, elapsedMs);
  }

  return {
    averageMs: totalMs / measuredRuns,
    maxMs,
    minMs,
  };
}

function requireModel(
  model: TfliteModel | undefined,
  name: string,
): TfliteModel {
  if (model == null) {
    throw new Error(`Expected ${name} model to be loaded`);
  }
  return model;
}

describe('react-native-fast-tflite (harness)', () => {
  describe('bundled EfficientDet model', () => {
    let model: TfliteModel;

    beforeAll(async () => {
      model = await loadTensorflowModel(MODEL_ASSET, CPU_DELEGATES);
    });

    it('uses no hardware delegates when an empty list is passed', () => {
      expect(model.delegates).toEqual([]);
    });

    it('exposes non-empty input and output tensor metadata', () => {
      expect(model.inputs.length).toBeGreaterThanOrEqual(1);
      expect(model.outputs.length).toBeGreaterThanOrEqual(1);
      for (const t of [...model.inputs, ...model.outputs]) {
        expect(t.name.length).toBeGreaterThan(0);
        expect(t.dataType).not.toBe('none');
        expect(t.shape.length).toBeGreaterThan(0);
        expect(t.shape.every(d => d > 0)).toBe(true);
      }
    });

    it('runSync returns one buffer per output with expected byte sizes', () => {
      const inputBuf = zeroedInputBuffer(firstInputTensor(model));
      const outputs = model.runSync([inputBuf]);
      expect(outputs).toHaveLength(model.outputs.length);
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs[i];
        const spec = model.outputs[i];
        expect(out).toBeDefined();
        expect(spec).toBeDefined();
        expect(out!.byteLength).toBe(tensorByteLength(spec!));
      }
    });

    it('runSync rejects the wrong number of input buffers', () => {
      expect(() => model.runSync([])).toThrow(/input array size/i);
    });

    it('every float32 output has only finite values for a zero input', () => {
      const inputBuf = zeroedInputBuffer(firstInputTensor(model));
      const outputs = model.runSync([inputBuf]);
      for (let i = 0; i < model.outputs.length; i++) {
        const buf = outputs[i];
        const spec = model.outputs[i];
        expect(buf).toBeDefined();
        expect(spec).toBeDefined();
        expectAllFloat32Finite(buf!, spec!);
      }
    });

    it('runSync is deterministic for a fixed zero input', () => {
      const inputBuf = zeroedInputBuffer(firstInputTensor(model));
      const first = model.runSync([inputBuf]);
      const second = model.runSync([inputBuf]);
      expect(first).toHaveLength(second.length);
      for (let i = 0; i < first.length; i++) {
        const a = first[i];
        const b = second[i];
        expect(a).toBeDefined();
        expect(b).toBeDefined();
        expect(buffersEqual(a!, b!)).toBe(true);
      }
    });

    it('is deterministic for a fixed non-zero uint8 input when the tensor is uint8', () => {
      const input = firstInputTensor(model);
      if (input.dataType !== 'uint8') {
        return;
      }
      const buf = filledInputBuffer(input, 0x7f);
      const a = model.runSync([buf]);
      const b = model.runSync([buf]);
      expect(a).toHaveLength(b.length);
      for (let i = 0; i < a.length; i++) {
        expect(buffersEqual(a[i]!, b[i]!)).toBe(true);
      }
    });

    it('loading the same asset again yields the same tensor layout', async () => {
      const other = await loadTensorflowModel(MODEL_ASSET, CPU_DELEGATES);
      expect(tensorSpecsMatch(model.inputs, other.inputs)).toBe(true);
      expect(tensorSpecsMatch(model.outputs, other.outputs)).toBe(true);
      expect(other.delegates).toEqual([]);
    });

    it('two independently loaded models produce identical outputs for the same input', async () => {
      const other = await loadTensorflowModel(MODEL_ASSET, CPU_DELEGATES);
      const inputBuf = zeroedInputBuffer(firstInputTensor(model));
      const a = model.runSync([inputBuf]);
      const b = other.runSync([inputBuf]);
      expect(a).toHaveLength(b.length);
      for (let i = 0; i < a.length; i++) {
        expect(buffersEqual(a[i]!, b[i]!)).toBe(true);
      }
    });

    it('matches the num_detections read when layout is unchanged', () => {
      if (model.outputs.length <= 3) {
        return;
      }
      const meta = model.outputs[3];
      if (meta == null || meta.dataType !== 'float32') {
        return;
      }
      const inputBuf = zeroedInputBuffer(firstInputTensor(model));
      const result = model.runSync([inputBuf]);
      const detBuf = result[3];
      expect(detBuf).toBeDefined();
      const numDetections = new Float32Array(detBuf!)[0] ?? 0;
      expect(Number.isFinite(numDetections)).toBe(true);
      expect(numDetections).toBeGreaterThanOrEqual(0);
    });
  });

  describe('google-quant MobileNet classifier', () => {
    let model: TfliteModel;

    beforeAll(async () => {
      model = await loadTensorflowModel(QUANT_MODEL_ASSET, CPU_DELEGATES);
    });

    it('has expected input shape [1, 224, 224, 3] with uint8 data type', () => {
      const input = firstInputTensor(model);
      expect(input.dataType).toBe('uint8');
      expect(input.shape).toEqual([1, 224, 224, 3]);
    });

    it('has a single output with 1001 classes', () => {
      expect(model.outputs).toHaveLength(1);
      const output = model.outputs[0]!;
      expect(output.dataType).toBe('uint8');
      expect(output.shape).toEqual([1, 1001]);
    });

    it('runs inference with a panda-colored synthetic image', () => {
      const input = firstInputTensor(model);
      const size = input.shape[1]; // 224
      const buf = new ArrayBuffer(tensorByteLength(input));
      const pixels = new Uint8Array(buf);

      // Paint a black-and-white panda-like pattern:
      // top half white (face), bottom half black (body),
      // with black "eye patches" in the upper quadrants.
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const idx = (y * size + x) * 3;
          const isTopHalf = y < size / 2;
          const isEyePatch =
            isTopHalf &&
            y > size * 0.2 &&
            y < size * 0.4 &&
            ((x > size * 0.15 && x < size * 0.35) ||
              (x > size * 0.65 && x < size * 0.85));

          let v: number;
          if (isEyePatch) {
            v = 20; // dark eye patches
          } else if (isTopHalf) {
            v = 240; // white face
          } else {
            v = 30; // dark body
          }
          pixels[idx] = v;
          pixels[idx + 1] = v;
          pixels[idx + 2] = v;
        }
      }

      const outputs = model.runSync([buf]);
      expect(outputs).toHaveLength(1);
      const scores = new Uint8Array(outputs[0]!);
      expect(scores.length).toBe(1001);

      // At least one class should have a non-zero score
      const maxScore = Math.max(...Array.from(scores));
      expect(maxScore).toBeGreaterThan(0);
    });

    it('top prediction has higher confidence than the average', () => {
      const input = firstInputTensor(model);
      const buf = filledInputBuffer(input, 0x80); // mid-gray image
      const outputs = model.runSync([buf]);
      const scores = Array.from(new Uint8Array(outputs[0]!));
      const max = Math.max(...scores);
      const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
      expect(max).toBeGreaterThan(avg);
    });

    it('is deterministic for the panda-colored input', () => {
      const input = firstInputTensor(model);
      const buf = filledInputBuffer(input, 0x7f);
      const a = model.runSync([buf]);
      const b = model.runSync([buf]);
      expect(buffersEqual(a[0]!, b[0]!)).toBe(true);
    });

    it('classifies a real panda photo as giant panda (top-5)', async () => {
      const input = firstInputTensor(model);
      const size = input.shape[1]; // 224

      const rgb = await fetchImageAsRGB(PANDA_IMAGE_URL, size);
      const outputs = model.runSync([rgb.buffer]);
      const scores = new Uint8Array(outputs[0]!);

      // Find top-5 class indices
      const indexed = Array.from(scores).map((score, i) => ({ i, score }));
      indexed.sort((a, b) => b.score - a.score);
      const top5Indices = indexed.slice(0, 5).map(x => x.i);

      expect(top5Indices).toContain(GIANT_PANDA_INDEX);
    });

    it('runSync rejects an input buffer that is too small', () => {
      const input = firstInputTensor(model);
      const tooSmall = new ArrayBuffer(tensorByteLength(input) - 1);
      expect(() => model.runSync([tooSmall])).toThrow(
        /does not match input tensor/i,
      );
    });

    it('runSync rejects an input buffer that is too large', () => {
      const input = firstInputTensor(model);
      const tooLarge = new ArrayBuffer(tensorByteLength(input) + 1);
      expect(() => model.runSync([tooLarge])).toThrow(
        /does not match input tensor/i,
      );
    });

    // Regression guard: a wrong-sized buffer used to be dropped silently, so
    // inference ran on whatever was left in the tensor from the previous call
    // and returned plausible-looking (but stale) scores instead of throwing.
    it('does not silently run inference on the previous input when the buffer size is wrong', () => {
      const input = firstInputTensor(model);
      const byteLength = tensorByteLength(input);

      // Prime the input tensor with a known image and keep a copy of its scores.
      const primed = copyOf(
        model.runSync([filledInputBuffer(input, 0x10)])[0]!,
      );

      // A visibly different image, but one byte short.
      const wrongSized = new Uint8Array(byteLength - 1).fill(0xf0)
        .buffer as ArrayBuffer;

      let threw = false;
      let reusedStaleInput = false;
      try {
        const outputs = model.runSync([wrongSized]);
        reusedStaleInput = buffersEqual(copyOf(outputs[0]!), primed);
      } catch {
        threw = true;
      }

      expect(reusedStaleInput).toBe(false);
      expect(threw).toBe(true);

      // The rejection must be specific to the size mismatch: a correctly sized
      // buffer with the same content still runs and reaches the tensor, so the
      // scores differ from the primed ones.
      const correctlySized = filledInputBuffer(input, 0xf0);
      const after = copyOf(model.runSync([correctlySized])[0]!);
      expect(after.byteLength).toBe(tensorByteLength(model.outputs[0]!));
      expect(buffersEqual(after, primed)).toBe(false);
    });
  });

  describe('android xnnpack delegate', () => {
    let cpuModel: TfliteModel | undefined;
    let xnnpackModel: TfliteModel | undefined;

    beforeAll(async () => {
      if (Platform.OS !== 'android') {
        return;
      }
      cpuModel = await loadTensorflowModel(QUANT_MODEL_ASSET, CPU_DELEGATES);
      xnnpackModel = await loadTensorflowModel(
        QUANT_MODEL_ASSET,
        XNNPACK_DELEGATES,
      );
    });

    it('loads only when explicitly requested on android', () => {
      if (Platform.OS !== 'android') {
        return;
      }
      expect(requireModel(cpuModel, 'cpu').delegates).toEqual([]);
      expect(requireModel(xnnpackModel, 'xnnpack').delegates).toEqual([
        'xnnpack',
      ]);
    });

    it('matches the default CPU tensor layout', () => {
      if (Platform.OS !== 'android') {
        return;
      }
      const cpu = requireModel(cpuModel, 'cpu');
      const xnnpack = requireModel(xnnpackModel, 'xnnpack');
      expect(tensorSpecsMatch(cpu.inputs, xnnpack.inputs)).toBe(true);
      expect(tensorSpecsMatch(cpu.outputs, xnnpack.outputs)).toBe(true);
    });

    it('runs inference with output sizes matching the default CPU path', () => {
      if (Platform.OS !== 'android') {
        return;
      }
      const cpu = requireModel(cpuModel, 'cpu');
      const xnnpack = requireModel(xnnpackModel, 'xnnpack');
      const input = firstInputTensor(cpu);
      const inputBuffer = filledInputBuffer(input, 0x80);
      const cpuOutputs = cpu.runSync([inputBuffer]);
      const xnnpackOutputs = xnnpack.runSync([inputBuffer]);
      expect(cpuOutputs).toHaveLength(xnnpackOutputs.length);
      for (let i = 0; i < cpuOutputs.length; i++) {
        expect(xnnpackOutputs[i]!.byteLength).toBe(cpuOutputs[i]!.byteLength);
      }
    });

    it('rejects xnnpack mixed with hardware delegates', async () => {
      if (Platform.OS !== 'android') {
        return;
      }
      await expect(
        loadTensorflowModel(QUANT_MODEL_ASSET, ['xnnpack', 'nnapi']),
      ).rejects.toThrow(/xnnpack delegate cannot be combined/i);
    });

    it('compares default CPU and xnnpack runSync timing', () => {
      if (Platform.OS !== 'android') {
        return;
      }
      const cpu = requireModel(cpuModel, 'cpu');
      const xnnpack = requireModel(xnnpackModel, 'xnnpack');
      const input = firstInputTensor(cpu);
      const cpuTiming = benchmarkRunSync(
        cpu,
        input,
        BENCHMARK_WARMUP_RUNS,
        BENCHMARK_RUNS,
      );
      const xnnpackTiming = benchmarkRunSync(
        xnnpack,
        input,
        BENCHMARK_WARMUP_RUNS,
        BENCHMARK_RUNS,
      );
      console.log(
        JSON.stringify({
          benchmark: 'tflite-cpu-vs-xnnpack',
          runs: BENCHMARK_RUNS,
          cpu: cpuTiming,
          xnnpack: xnnpackTiming,
        }),
      );
      expect(cpuTiming.averageMs).toBeGreaterThan(0);
      expect(xnnpackTiming.averageMs).toBeGreaterThan(0);
    });
  });
});
