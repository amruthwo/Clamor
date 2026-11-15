# Building Clamor on Asahi Fedora (M1) and pushing to GitHub

Target: `github.com/amruthwo/Clamor`. Your M1 Air is aarch64 Linux, so builds here are
native for `linux-aarch64`. Everything else (macOS/Metal, Windows, Android, x86_64, CUDA)
is built and tested by GitHub Actions on the runners in `.github/workflows/ci.yml`.

## 1. Toolchain

```bash
sudo dnf install -y gcc gcc-c++ clang clang-devel cmake ninja-build git python3 python3-pip \
    vulkan-loader-devel vulkan-tools mesa-vulkan-drivers glslc spirv-headers-devel openssl-devel pkgconf-pkg-config gh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
pip install --user numpy gguf        # for tools/ and one test
```

Check Vulkan (Honeykrisp) sees the GPU:

```bash
vulkaninfo --summary | grep -E "deviceName|apiVersion"
```

If that fails, update first: `sudo dnf upgrade --refresh` and reboot.

## 2. Get the code in

Unpack the tarball and push it as the first commit:

```bash
tar xzf clamor-v0.3.tar.gz && cd clamor
git init -b main
git add -A && git commit -m "Initial import: muster, node, transport, ggml engine scaffold"
gh auth login                         # GitHub.com → HTTPS → browser login
git remote add origin https://github.com/amruthwo/Clamor.git
git push -u origin main               # the placeholder repo is empty, so this is a plain push
```

(If the placeholder has a README commit, use `git pull --rebase origin main --allow-unrelated-histories` first.)

## 3. Build and test locally

```bash
cargo test --workspace --release                       # CPU only
cargo run -p clamor-engine-ggml --release --example devices
cargo run -p clamor-engine-ggml --release --features vulkan --example devices   # + Honeykrisp
```

First ggml build takes a few minutes (Vulkan shaders compile via glslc). With `--features vulkan` you should see
two devices: `CPU` and `Vulkan0` (Apple M1 Honeykrisp). If matmul on Vulkan0 fails, that is a real finding: capture
the output and open an issue.

Set `CLAMOR_GGML_NATIVE=1` for a locally tuned CPU build; leave it unset for anything you will copy to another machine.

## 4. Iterate with CI

```bash
git checkout -b feat/x && git push -u origin feat/x && gh pr create --fill
gh run watch                          # follow the matrix
```

The CI jobs marked build-only (Vulkan, Android, CUDA) prove it compiles, not that it runs: you cannot test those on
Asahi or on hosted runners. Run `examples/devices` by hand on a real NVIDIA/Android device when you have one.

## 5. What you cannot do on this machine

- Metal: needs macOS. CI covers build and CPU/Metal test on `macos-15` (Apple silicon runner; Metal availability in
  hosted VMs can be limited, so a failure in the Metal matmul there may be the runner, not the code).
- CUDA: needs an NVIDIA machine to run. Android: build via CI (no official NDK for aarch64 Linux).
