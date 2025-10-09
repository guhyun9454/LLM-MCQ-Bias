from __future__ import annotations
import argparse
import subprocess
import sys
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_txt", type=str, default="models.txt",
                        help="모델 리스트가 줄 단위로 적힌 파일 경로")
    parser.add_argument("--eval_names", type=str, nargs='+', default=["arc,0,cyclic", "csqa,0,cyclic"],
                        help="평가 작업 이름 리스트. 예) arc,0,cyclic csqa,0,cyclic mmlu,0,cyclic")
    parser.add_argument("--data_root", type=str, default="data",
                        help="데이터 루트 디렉터리 (data_{task}들이 있는 곳)")
    parser.add_argument("--ko", action="store_true",
                        help="한국어 CSV 사용 ( *_dev.ko.csv / *_test.ko.csv )")
    parser.add_argument("--prompt_lang", type=str, choices=["en", "ko"], default="ko",
                        help="프롬프트 언어")
    parser.add_argument("--option_ids4", type=str, default="가,나,다,라",
                        help="4지선다 표기. 예) A,B,C,D 또는 가,나,다,라")
    parser.add_argument("--option_ids5", type=str, default="가,나,다,라,마",
                        help="5지선다 표기. 예) A,B,C,D,E 또는 가,나,다,라,마")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="파이썬 실행 파일 경로")
    parser.add_argument("--debias_fn", type=str, default="simple", choices=["simple", "full"],
                        help="debias 함수 선택 (perm 결과에는 full도 가능)")
    parser.add_argument("--debias_pct", type=float, default=1.0,
                        help="디바이어싱에 사용할 샘플 비율 (0~1). 1.0은 전체 사용")
    parser.add_argument("--dry_run", action="store_true", help="명령만 출력하고 실행하지 않음")
    return parser.parse_args()


def read_models(models_txt: Path) -> list[str]:
    models: list[str] = []
    if not models_txt.exists():
        raise FileNotFoundError(f"모델 목록 파일을 찾을 수 없습니다: {models_txt}")
    for line in models_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        models.append(line)
    if not models:
        raise ValueError("models.txt에 유효한 모델 항목이 없습니다.")
    return models


def build_eval_cmd(python: str, model: str, eval_names: list[str], data_root: str,
                   ko: bool, prompt_lang: str, option_ids4: str | None, option_ids5: str | None) -> list[str]:
    cmd = [
        python,
        "code/eval_clm.py",
        "--pretrained_model_path", model,
        "--eval_names", *eval_names,
        "--data_root", data_root,
        "--prompt_lang", prompt_lang,
    ]
    if ko:
        cmd.append("--ko")
    if option_ids4:
        cmd.extend(["--option_ids4", option_ids4])
    if option_ids5:
        cmd.extend(["--option_ids5", option_ids5])
    return cmd


def model_name_from_path(model_path: str) -> str:
    return model_path.rstrip("/").split("/")[-1]


def extract_tasks(eval_names: list[str]) -> list[str]:
    tasks = []
    for e in eval_names:
        parts = e.split(',')
        if len(parts) == 0:
            continue
        t = parts[0].strip()
        if t not in tasks:
            tasks.append(t)
    return tasks


def expected_result_dir(task: str, eval_name: str, model_name: str) -> Path:
    parts = eval_name.split(',')
    num_few_shot = int(parts[1]) if len(parts) > 1 else 0
    setting = parts[2] if len(parts) > 2 and parts[2] else None
    base = Path(f"results/{task}/{num_few_shot}s_{model_name}/{task}")
    if setting:
        base = base.with_name(f"{task}_{setting}")
    return base


def build_debias_cmds(python: str, debias_fn: str, eval_names: list[str], model: str,
                      option_ids4: str | None, option_ids5: str | None,
                      debias_pct: float) -> list[list[str]]:
    model_name = model_name_from_path(model)
    cmds: list[list[str]] = []

    for e in eval_names:
        task = e.split(',')[0]
        load_path = str(expected_result_dir(task, e, model_name))
        if not os.path.isdir(load_path):
            # 경로가 없는 경우 스킵 (평가가 아직 안 끝난 경우)
            continue

        cmd = [
            python, "code/debias_base.py",
            "--task", task,
            "--debias_fn", debias_fn,
            "--load_paths", load_path,
        ]
        # option ids
        if task == "csqa":
            if option_ids5:
                cmd.extend(["--option_ids5", option_ids5])
        else:
            if option_ids4:
                cmd.extend(["--option_ids4", option_ids4])

        # debias_pct는 환경변수로 전달해서 utils._index_samples가 부분 샘플링 하도록 유도
        # (eval 단계에는 영향 없음. debias_base는 저장 시 전체를 필요로 하지만,
        #  utils.eval_all_samples가 사용하는 _index_samples는 환경변수로 제어되지 않음)
        # 따라서 debias 단계에서 jsonl을 부분 샘플링해 별도 임시 디렉터리로 복사하는 것이 안전하다.
        # 이 파일에서는 명령 실행 직전에 처리한다.
        cmds.append(["__DEBIAS__", f"{debias_pct}"] + cmd)

    return cmds


def run_cmd(cmd: list[str], dry_run: bool = False) -> int:
    blue = "\033[94m"
    reset = "\033[0m"
    print(f"{blue}Command: {' '.join(cmd)}{reset}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    return proc.returncode


def copy_subset_records(src_dir: Path, dst_dir: Path, pct: float) -> None:
    import json
    import math
    os.makedirs(dst_dir, exist_ok=True)
    for file in sorted(os.listdir(src_dir)):
        if not file.endswith('.jsonl'):
            continue
        src = src_dir / file
        dst = dst_dir / file
        with open(src, 'r') as f:
            lines = [json.loads(line) for line in f]
        results = [e for e in lines if e.get('type') == 'result']
        metrics = [e for e in lines if e.get('type') == 'metric']
        if pct >= 1.0:
            subset = results
        else:
            n = max(1, math.floor(len(results) * max(0.0, min(1.0, pct))))
            subset = results[:n]
        with open(dst, 'w') as f:
            for r in subset + metrics:
                f.write(json.dumps(r) + '\n')


def main() -> None:
    args = parse_args()
    models = read_models(Path(args.models_txt))

    blue = "\033[94m"
    reset = "\033[0m"

    for model in models:
        print(f"{blue}Model: {model}{reset}")

        # 1) 평가 실행
        eval_cmd = build_eval_cmd(
            args.python, model, args.eval_names, args.data_root,
            args.ko, args.prompt_lang, args.option_ids4, args.option_ids5,
        )
        rc = run_cmd(eval_cmd, dry_run=args.dry_run)
        if rc != 0:
            print(f"[FAIL] eval model={model} (returncode={rc})")
            continue

        # 2) debias 실행 (필요 시 부분 샘플링)
        debias_cmds = build_debias_cmds(
            args.python, args.debias_fn, args.eval_names, model,
            args.option_ids4, args.option_ids5, args.debias_pct,
        )
        for dcmd in debias_cmds:
            assert dcmd[0] == "__DEBIAS__"
            pct = float(dcmd[1])
            base_cmd = dcmd[2:]

            # 부분 샘플링 처리: load_path를 임시 디렉터리로 복제 후 앞부분만 사용
            load_path_idx = base_cmd.index("--load_paths") + 1
            orig_load_path = Path(base_cmd[load_path_idx])
            if pct >= 1.0:
                # 그대로 실행
                rc2 = run_cmd(base_cmd, dry_run=args.dry_run)
                if rc2 != 0:
                    print(f"[FAIL] debias model={model} (returncode={rc2})")
                continue

            tmp_dir = orig_load_path.parent / (orig_load_path.name + f"_subset_{int(pct*100)}")
            copy_subset_records(orig_load_path, tmp_dir, pct)

            # 명령의 load_paths를 임시 경로로 치환
            base_cmd[load_path_idx] = str(tmp_dir)
            rc2 = run_cmd(base_cmd, dry_run=args.dry_run)
            if rc2 != 0:
                print(f"[FAIL] debias model={model} (returncode={rc2})")


if __name__ == "__main__":
    main()



