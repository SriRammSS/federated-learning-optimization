from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from flopt.mimic import MimicConfig,preprocess_mimic


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",default="data")
    parser.add_argument("--out",default="outputs/full_mimic_iv")
    parser.add_argument("--db-path",default="data/mimic_cache/mimic_iv.duckdb")
    parser.add_argument("--hours",type=int,default=24)
    parser.add_argument("--top-chart",type=int,default=50)
    parser.add_argument("--top-labs",type=int,default=50)
    parser.add_argument("--top-inputs",type=int,default=30)
    parser.add_argument("--top-outputs",type=int,default=20)
    parser.add_argument("--top-procedures",type=int,default=25)
    parser.add_argument("--top-rx",type=int,default=30)
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--threads",type=int,default=12)
    parser.add_argument("--memory-limit",default="36GB")
    args=parser.parse_args()
    cfg=MimicConfig(
        data_dir=Path(args.data_dir),
        out=Path(args.out),
        db_path=Path(args.db_path),
        hours=args.hours,
        top_chart=args.top_chart,
        top_labs=args.top_labs,
        top_inputs=args.top_inputs,
        top_outputs=args.top_outputs,
        top_procedures=args.top_procedures,
        top_rx=args.top_rx,
        seed=args.seed,
        threads=args.threads,
        memory_limit=args.memory_limit,
    )
    meta=preprocess_mimic(cfg)
    print(f"wrote MIMIC-IV preprocessing and EDA outputs to {meta['mimic_root']} -> {cfg.out}")


if __name__=="__main__":
    main()
