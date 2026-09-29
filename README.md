# CrackBit-Random Puzzle 71
CUDA-accelerated Bitcoin key generator with Philox RNG, Feistel permutation, random search, and checkpoint support (v2/v3/v4/v5).
Bitcoin key generator (CUDA) – Philox, Feistel, secp256k1, SHA256, RIPEMD160, batch inversion, checkpoint. GPU random search in custom ranges.


========================================
 Bitcoin GPU key generator (v2/v3/v4/v5)
   [OPT: mixed-add + 5bit-windows + preset]
========================================
  Mod: v4  (Philox + random pur)
  -- rkey = 100,000,000,000 (refresh start points)
========================================
GPU         : NVIDIA GeForce RTX 3060 CC 8.6  (28 SM-uri)
Block size  : 256 thread-uri/bloc
Grid size   : 25,600 blocuri [gpux override/preset]
Keys/iter   : 26,214,400
KPT         : 4 keys/thread
Occupancy   : feistel=2/SM, random=2/SM  -> folosit: 2/SM
Waves       : 64 (auto-batch)
I/O         : normal
CSPRNG      : Philox4x32-10 (counter-based)
RKEY        : 100,000,000,000
Scalar mul  : 5-bit comb x 15 windows (13 add)
Point add   : Mixed Jacobian+Affine (11 mul)
Batch inv   : warp-parallel (shfl_sync, 32 elem/inv)
Range       : 20000000:3fffffff
Range size  : 0x000000000020000000 chei
Target addr : 1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps
Output file : found_keys.txt  (append mode)


=== SELF TEST ===
CPU pubkey  : 020AB331BE12B1229ECE67D4C63240B30B348AF82E7C01C332DEC983E31A2722FD
CPU address : 14SLfcKMypXcWNGYQVw4JEXnee2CH6K2wM
Expected    : 14SLfcKMypXcWNGYQVw4JEXnee2CH6K2wM
SELF TEST OK (CPU-only check)
Cursor: 0x000000000000000000 (0.0000%) | 784270.6 keys/s | Total: 52,428,800 | Batches: 2

==================================================
 *** MATCH FOUND ***
==================================================
 Private key (32B hex): 000000000000000000000000000000000000000000000000000000003D94CD64
 Private key (9B  hex): 00000000003D94CD64
 Address              : 1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps
 Saved to             : found_keys.txt
==================================================
Search finished | Total: 78,643,200 | Avg: 784259.34 keys/s
Cheile gasite au fost salvate in: found_keys.txt

---

---

## ✨ Features

- **4 search modes** (`v2`, `v3`, `v4`, `v5`) selectable from CLI
- **Philox4x32-10** counter-based CSPRNG (GPU-side, no host bottlenecks)
- **Feistel permutation** for 100% coverage without replacement (v3)
- **Fixed-base comb scalar multiplication** (4-bit × 18 windows)
- **Warp-parallel batch inversion** using `__shfl_sync`
- **Checkpoint / resume** for long-running scans (v5)
- **Auto-grid tuning** based on SM count and kernel occupancy
- **KPT (keys per thread)** templated: `1, 2, 4, 8, 16`
- **Append-only output** — found keys are never lost (`found_keys.txt`)

---

## 🧩 Modes

| Mode | Strategy | Replacement | Coverage | Checkpoint |
|------|-----------|-------------|----------|------------|
| `v2` | Philox + cursor + sub-windows | No | Partial (bounded) | No |
| `v3` | Feistel permutation **(default)** | No | **100%** | No |
| `v4` | Philox random over full range | Yes | Infinite | No |
| `v5` | Block scan + random + checkpoint | Yes | Bounded by block | **Yes** |

---

## 🛠️ Build

### Prerequisites

```bash
sudo apt install build-essential libssl-dev libsecp256k1-dev

Compile
Turing (RTX 20xx / GTX 16xx / T4):
nvcc -O3 -arch=sm_75 -std=c++14 s6.cu -o CrackBit \
     -lsecp256k1 -lcrypto -lpthread

Architecture cheat-sheet
GPU family	Flag
Pascal (GTX 10xx, P100)	-arch=sm_60 / sm_61
Volta (V100, Titan V)	-arch=sm_70
Turing (RTX 20xx, T4)	-arch=sm_75
Ampere (RTX 30xx, A100)	-arch=sm_80 / sm_86
Ada (RTX 40xx)	-arch=sm_89
Hopper (H100)	-arch=sm_90

🚀 Usage

./wallet_cuda [-v2|-v3|-v4|-v5] [opts] START:END TARGET [BATCH]
START:END — hex range (up to 72 bits, i.e. 18 hex chars each)

TARGET — P2PKH address (Base58, mainnet)

BATCH — optional batch size (default: auto)

Options
Flag	Description
-v2 | --v2	Philox + cursor + sub-windows
-v3 | --v3	Feistel permutation (default)
-v4 | --v4	Philox pure random over full range
-v5 | --v5	Block scan + random + checkpoint
-kpt N	Keys per thread: 1, 2, 4, 8, 16 (default 4)
-blocks N	Override grid size
-gpux N,M	2D grid (Rotor-Cuda style, e.g. 18000,512)
-rkey N	Refresh start points every N keys (0 = default for v4/v5)
-step N	For v3: only test positions multiple of N
-block N	v5 block size (default 1e9)
-sample N	v5 keys per block (default 1e7)
-reset	Clear v5 checkpoint
-q | --quiet	Reduce progress I/O
-bench	Benchmark KPT 4, 8, 16 (30s each)
-h | --help	Help
