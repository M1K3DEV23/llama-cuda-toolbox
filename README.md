# llama-cuda-toolbox

Compila e instala [llama.cpp](https://github.com/ggml-org/llama.cpp) con **CUDA** en Linux usando **Podman** y las imágenes oficiales `nvidia/cuda`. Un solo script (`llama-cuda`): clona fuentes, compila en contenedor con GPU, exporta binarios al host y configura `PATH` / `LD_LIBRARY_PATH`.

Pensado para **Fedora** (u otra distro con Podman rootless + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)), sin ensuciar el sistema con toolkits CUDA en el host.

## Características

- Build reproducible en contenedor `nvidia/cuda:*-devel-ubuntu24.04`
- Perfiles de CUDA por **versión de toolkit** (`toolkit-13.0`, `toolkit-12.8`, `stable`, `latest`)
- Flags CMake configurables: Blackwell `120`, FlashAttention all quants, CUDA graphs, `GGML_NATIVE`, OpenSSL (`-hf` HTTPS)
- Git inteligente: `update` solo recompila si hay commits o cambió `config`
- Instalación en `~/.local/bin/llama-cuda` + libs runtime (`libcudart`, `cublas`, **NCCL**)
- `uninstall` / `uninstall --purge`

## Requisitos

| Componente | Notas |
|------------|--------|
| Linux x86_64 | Probado en Fedora 44+ |
| [Podman](https://podman.io/) | `sudo dnf install -y podman` |
| NVIDIA driver + GPU | Driver reciente (p. ej. 550+ para Blackwell) |
| NVIDIA Container Toolkit | GPU visible dentro del contenedor (`nvidia.com/gpu=all`) |
| `git` | Clone/fetch en el host |

Comprueba GPU en contenedor:

```bash
podman run --rm --device nvidia.com/gpu=all docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04 nvidia-smi
```

## Inicio rápido

```bash
git clone https://github.com/M1K3DEV23/llama-cuda-toolbox.git
cd llama-cuda-toolbox
cp config.example config
# Edita config si cambias perfil o flags
./llama-cuda build --profile toolkit-13.0
source ~/.bashrc
llama-cli --list-devices
```

Deberías ver tu GPU, por ejemplo:

```text
CUDA0: NVIDIA GeForce RTX 5080 (…)
```

## Perfiles de compilación

El **perfil** elige la imagen Podman (versión de CUDA / nvcc), no solo flags CMake.

| Perfil | Imagen | Uso |
|--------|--------|-----|
| `toolkit-13.0` | `cuda:13.0.0-devel-ubuntu24.04` | Toolkit CUDA **13.0** explícito |
| `toolkit-12.8` | `cuda:12.8.0-devel-ubuntu24.04` | Toolkit **12.8** |
| `stable` | `cuda:13.2.x` (fallback 13.1 / 13.0) | Misma imagen reciente + `-Xcicc -O1` (cuants IQ en 13.2+) |
| `latest` | Imagen más nueva disponible | Máximo rendimiento; posible gibberish IQ en 13.2+ |

```bash
./llama-cuda rebuild --profile toolkit-13.0
./llama-cuda rebuild --profile stable
```

## Comandos

```bash
./llama-cuda build              # instala/compila si hace falta
./llama-cuda update             # fetch + compila solo si hay cambios (recomendado)
./llama-cuda update --check     # exit 2 si hay actualización pendiente
./llama-cuda rebuild            # borra build/ y recompila todo
./llama-cuda rebuild --force    # reset git duro + rebuild
./llama-cuda status             # git, perfil CUDA, último build
./llama-cuda pull               # solo descarga imagen Podman
./llama-cuda rmi-images         # borra imágenes nvidia/cuda (13.x, 12.8); no toca binarios ni fuentes
./llama-cuda env                # export PATH / LD_LIBRARY_PATH
./llama-cuda uninstall          # quita binarios y entradas en .bashrc
./llama-cuda uninstall --purge  # + fuentes ~/.cache/llama-cpp-src + imágenes
./llama-cuda clean              # solo borra directorio build/
```

Opciones: `--profile`, `--force`, `--check`, `--branch`.

## Configuración (`config`)

```bash
cp config.example config
```

| Variable | Descripción |
|----------|-------------|
| `CUDA_BUILD_PROFILE` | `toolkit-13.0`, `stable`, `latest`, `toolkit-12.8` |
| `CUDA_IMAGE_TAG` | Tag base para perfiles `stable` / `latest` (p. ej. `13.2.0`) |
| `CUDA_ARCH` | `CMAKE_CUDA_ARCHITECTURES` (p. ej. `120` en RTX 50xx / Blackwell) |
| `GGML_CUDA_FA_ALL_QUANTS` | `ON` = FlashAttention todas las cuants (compila más lento) |
| `GGML_CUDA_GRAPHS` | `ON` = CUDA graphs en server |
| `GGML_NATIVE` | `ON` = CPU ggml con `-march=native` |
| `LLAMA_OPENSSL` | `ON` = HTTPS para `llama-server -hf` |

Equivalente CMake por defecto en `config.example`:

```cmake
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_NATIVE=ON \
  -DLLAMA_OPENSSL=ON
```

**Nota Blackwell:** si pasas `CUDA_ARCH=120`, CMake de llama.cpp puede promover a `120a` para kernels FP4; es comportamiento upstream, no del script.

## Rutas en el host

| Qué | Ruta |
|-----|------|
| Script | `./llama-cuda` |
| Config activo | `./config` (no versionado; ver `.gitignore`) |
| Fuentes llama.cpp | `~/.cache/llama-cpp-src` |
| Build | `~/.cache/llama-cpp-src/build` |
| Binarios | `~/.local/bin/llama-cuda/` |
| Librerías CUDA/NCCL | `~/.local/lib/llama-cuda/` |

## Uso tras instalar

```bash
# Modelo desde Hugging Face (requiere LLAMA_OPENSSL=ON)
llama-cli -hf org/repo:quant -p "Hola" -n 64

# Servidor API
llama-server -hf org/repo:quant --host 127.0.0.1 --port 8080
```

Modelos descargados suelen quedar en `~/.cache/huggingface` (no los borra `uninstall`).

## Problemas frecuentes

| Síntoma | Solución |
|---------|----------|
| `libnccl.so.2: cannot open shared object file` | Reinstala libs: `./llama-cuda build` o copia `libnccl` desde la imagen (ver script `install_bins_and_libs`) |
| GPU no aparece en contenedor | Revisa NVIDIA Container Toolkit y `podman run … nvidia-smi` |
| `Permission denied` al desinstalar | Binarios creados como root; el script usa `sudo rm` si hace falta |
| Rama divergió | `./llama-cuda update --force` |
| Puerto ocupado | `pkill -f llama-server` o otro `--port` |

## Hardware de referencia

Documentado y probado con:

- **GPU:** NVIDIA GeForce RTX 5080 (Blackwell, compute `12.0` / arch `120`)
- **CPU:** AMD Ryzen 9800X3D (`GGML_NATIVE=ON`)

Otras GPUs NVIDIA con driver + CDI deberían funcionar ajustando `CUDA_ARCH` y el perfil CUDA.

## Licencia

MIT — ver [LICENSE](LICENSE).

llama.cpp es proyecto aparte ([licencia propia](https://github.com/ggml-org/llama.cpp)); este repo solo distribuye el script de empaquetado/build.
