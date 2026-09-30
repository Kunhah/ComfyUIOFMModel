"""Pick the right vast.ai GPU for each job automatically.

Each job has hard requirements (VRAM, GPU generation, disk) and every offer that meets them is
ranked by estimated total cost, not by hourly price: you pay while the models download and while
the job runs, so a fast card on a fast line often beats a cheap slow one.

    estimated cost = $/hr x (download hours + work hours x reference speed / this GPU's speed)

Speed is vast.ai's `dlperf` score; the work-hour estimates were measured on an RTX 5090.

Requirements, and why:
  train    Krea 2 RAW LoRA with Ostris AI Toolkit (quantized, batch 1, gradient checkpointing).
           Needs ~30 GB VRAM, so 32 GB+ cards; bf16, so Ampere or newer (compute capability 8.0+).
  krea2    Krea 2 Turbo fp8 in ComfyUI (Workflows 07-10, 09b, 12). Fits in 24 GB; Ampere or newer,
           older cards lack bf16 and are too slow to be worth their lower price.
  minimax  MiniMax H3 (Workflow 11) plus Krea 2. The H3 text encoder ships in nvfp4, which only
           Blackwell runs (compute capability 10.0+: RTX 5090, RTX PRO 4000/5000/6000, B200), and
           H3 needs a 32 GB card. ~45 GB more to download, so a bigger disk.
"""
from __future__ import annotations

from dataclasses import dataclass

REF_DLPERF = 200  # RTX 5090, where the work-hour estimates below come from
EXCLUDED_NAMES = ("CMP",)  # mining cards: x1 PCIe, loading 20+ GB of weights takes forever


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    min_vram_gb: int
    min_compute_cap: int  # vast.ai's compute_cap: 800 = 8.0 (Ampere), 1000 = 10.0 (Blackwell)
    disk_gb: int
    max_price: float  # $/hr ceiling unless --max-price says otherwise
    download_gb: float  # what the pod pulls before it can work
    work_hours: float  # typical job length on the reference GPU


PROFILES = {
    "train": Profile("train", "Krea 2 LoRA training", min_vram_gb=31, min_compute_cap=800, disk_gb=150,
                     max_price=1.00, download_gb=62, work_hours=1.5),
    "krea2": Profile("krea2", "Krea 2 generation", min_vram_gb=23, min_compute_cap=800, disk_gb=60,
                     max_price=0.60, download_gb=20, work_hours=1.0),
    "minimax": Profile("minimax", "MiniMax H3 video + Krea 2", min_vram_gb=31, min_compute_cap=1000, disk_gb=200,
                       max_price=1.50, download_gb=65, work_hours=1.0),
}


def apply_defaults(args, profile: Profile) -> None:
    """Fill every search argument the user didn't pass from the job's profile. A GPU pinned with
    --gpu is the user's call, so the generation requirement is dropped unless --min-cc is given
    too (e.g. --with-minimax --gpu RTX_4090 runs H3 without nvfp4, several times slower)."""
    if getattr(args, "gpu", None) and getattr(args, "min_cc", None) is None:
        args.min_cc = 0
    for attr, value in (("vram", profile.min_vram_gb), ("disk", profile.disk_gb),
                        ("max_price", profile.max_price), ("min_cc", profile.min_compute_cap)):
        if getattr(args, attr, None) is None:
            setattr(args, attr, value)


def query(args) -> str:
    q = (f"num_gpus=1 gpu_ram>={args.vram} compute_cap>={args.min_cc} disk_space>={args.disk} "
         f"inet_down>={args.min_inet} rentable=true verified=true reliability>={args.min_reliability} "
         f"dph_total<={args.max_price} cuda_max_good>={args.min_cuda}")
    if args.gpu:  # pinned by hand: only that model, requirements still apply
        q = f"gpu_name={args.gpu} " + q
    return q


def estimate(offer: dict, profile: Profile, work_hours: float | None = None) -> tuple[float, float]:
    """(hours, dollars) for downloading the models and doing the job on this offer."""
    work = profile.work_hours if work_hours is None else work_hours
    download_h = profile.download_gb * 8 * 1000 / max(offer.get("inet_down") or 1, 1) / 3600
    run_h = work * REF_DLPERF / max(offer.get("dlperf") or 1, 1)
    hours = download_h + run_h
    return hours, hours * offer["dph_total"]


def rank(offers: list[dict], profile: Profile, work_hours: float | None = None) -> list[dict]:
    """Offers cheapest-overall first, each with _est_hours / _est_cost added."""
    kept = []
    for o in offers:
        if any(o.get("gpu_name", "").startswith(x) for x in EXCLUDED_NAMES):
            continue
        o["_est_hours"], o["_est_cost"] = estimate(o, profile, work_hours)
        kept.append(o)
    return sorted(kept, key=lambda o: (o["_est_cost"], -o.get("reliability2", 0)))


def describe(o: dict) -> str:
    return (f"{o['id']:>10}  {o['gpu_name']:<18} {o.get('gpu_ram', 0) / 1024:>3.0f}GB  ${o['dph_total']:.3f}/hr  "
            f"speed {o.get('dlperf', 0):>4.0f}  net {o.get('inet_down', 0):>5.0f}Mbps  "
            f"-> ~{o['_est_hours']:.1f}h ~${o['_est_cost']:.2f}")


def add_arguments(ap) -> None:
    """The search flags vast_pod.py and comfy_pod.py share. Blank ones come from the job profile."""
    ap.add_argument("--gpu", help="Pin one GPU model (e.g. RTX_4090). Default: pick automatically.")
    ap.add_argument("--vram", type=int, help="Minimum GPU RAM in GB. Default: what the job needs.")
    ap.add_argument("--min-cc", type=int, help="Minimum compute capability x100 (800 = Ampere, 1000 = Blackwell).")
    ap.add_argument("--disk", type=int, help="Disk GB. Default: what the job needs.")
    ap.add_argument("--max-price", type=float, help="Max $/hr. Default: per job.")
    ap.add_argument("--min-inet", type=int, default=300, help="Minimum download Mbps (you pay while models download).")
    ap.add_argument("--min-reliability", type=float, default=0.97)
    ap.add_argument("--min-cuda", type=float, default=12.9,
                    help="Host driver must support at least this CUDA version; otherwise the container "
                         "dies with 'CUDA error 804: forward compatibility' and you pay for an unusable GPU.")
    ap.add_argument("--limit", type=int, default=8, help="How many ranked offers to show.")
