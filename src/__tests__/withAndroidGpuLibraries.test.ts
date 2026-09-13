import type {
  AndroidConfig,
  ExportedConfig,
  ModProps,
} from '@expo/config-plugins'
import { withAndroidGpuLibraries } from '../expo-plugin/withAndroidGpuLibraries'

type AndroidManifest = AndroidConfig.Manifest.AndroidManifest
type ManifestUsesLibrary = AndroidConfig.Manifest.ManifestUsesLibrary
type ManifestApplicationWithNativeLibraries =
  AndroidConfig.Manifest.ManifestApplication & {
    'uses-native-library'?: ManifestUsesLibrary[]
  }

function createExpoConfig(): ExportedConfig {
  return { name: 'TfliteExample', slug: 'tflite-example' }
}

function createManifest(
  usesNativeLibrary?: ManifestUsesLibrary[]
): AndroidManifest {
  const application: ManifestApplicationWithNativeLibraries = {
    $: { 'android:name': '.MainApplication' },
  }
  if (usesNativeLibrary != null)
    application['uses-native-library'] = usesNativeLibrary

  return {
    manifest: {
      $: { 'xmlns:android': 'http://schemas.android.com/apk/res/android' },
      queries: [],
      application: [application],
    },
  }
}

/**
 * Runs the `withAndroidGpuLibraries` plugin's `android.manifest` mod against the
 * given manifest, the same way `expo prebuild` would, and returns the result.
 */
async function applyPlugin(
  manifest: AndroidManifest,
  enabledLibraries: boolean | string[]
): Promise<AndroidManifest> {
  const expoConfig = createExpoConfig()
  const config = withAndroidGpuLibraries(
    expoConfig,
    enabledLibraries
  ) as ExportedConfig
  const mod = config.mods?.android?.manifest
  if (mod == null)
    throw new Error('withAndroidGpuLibraries did not register a manifest mod!')

  const modRequest: ModProps<AndroidManifest> = {
    projectRoot: '/app',
    platformProjectRoot: '/app/android',
    modName: 'manifest',
    platform: 'android',
    introspect: false,
  }
  const result = await mod({
    ...config,
    modResults: manifest,
    modRequest: modRequest,
    modRawConfig: expoConfig,
  })

  return result.modResults
}

function getUsesNativeLibraries(
  manifest: AndroidManifest
): ManifestUsesLibrary[] {
  const application: ManifestApplicationWithNativeLibraries | undefined =
    manifest.manifest.application?.[0]
  if (application == null) throw new Error('No <application> in the manifest!')
  return application['uses-native-library'] ?? []
}

function getUsesNativeLibraryNames(manifest: AndroidManifest): string[] {
  return getUsesNativeLibraries(manifest).map((lib) => lib.$['android:name'])
}

describe('withAndroidGpuLibraries', () => {
  it('adds libOpenCL.so when enabled with `true`', async () => {
    const manifest = await applyPlugin(createManifest(), true)

    expect(getUsesNativeLibraryNames(manifest)).toEqual(['libOpenCL.so'])
  })

  it('marks the added libraries as not required', async () => {
    const manifest = await applyPlugin(createManifest(), true)

    expect(getUsesNativeLibraries(manifest)[0]?.$).toEqual({
      'android:name': 'libOpenCL.so',
      'android:required': false,
    })
  })

  it('adds libOpenCL.so alongside the explicitly listed libraries', async () => {
    const manifest = await applyPlugin(createManifest(), [
      'libOpenCL-pixel.so',
      'libGLES_mali.so',
    ])

    expect(getUsesNativeLibraryNames(manifest)).toEqual([
      'libOpenCL.so',
      'libOpenCL-pixel.so',
      'libGLES_mali.so',
    ])
  })

  it('does not duplicate entries when prebuild runs twice', async () => {
    const libraries = ['libOpenCL-pixel.so']
    const once = await applyPlugin(createManifest(), libraries)
    const twice = await applyPlugin(once, libraries)

    expect(getUsesNativeLibraryNames(twice)).toEqual([
      'libOpenCL.so',
      'libOpenCL-pixel.so',
    ])
  })

  it('keeps unrelated <uses-native-library> entries that are already present', async () => {
    const existing: ManifestUsesLibrary = {
      $: { 'android:name': 'libsomething-else.so', 'android:required': 'true' },
    }
    const manifest = await applyPlugin(createManifest([existing]), true)

    expect(getUsesNativeLibraryNames(manifest)).toEqual([
      'libsomething-else.so',
      'libOpenCL.so',
    ])
  })
})
