import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_txt", type=str, default="models.txt",
                        help="모델 리스트가 줄 단위로 적힌 파일 경로")
    parser.add_argument("--eval_names", type=str, nargs='+', default=["arc,0", "csqa,0"],
                        help="평가 작업 이름 리스트. 예) arc,0 csqa,0 mmlu,0")
    parser.add_argument("--data_root", type=str, default="../LLM-MCQ-Bias_data",
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


def main() -> None:
    args = parse_args()
    models = read_models(Path(args.models_txt))

    for model in models:
        print(f"[RUN] model={model}")
        cmd = [
            args.python,
            "code/eval_clm.py",
            "--pretrained_model_path", model,
            "--eval_names", *args.eval_names,
            "--data_root", args.data_root,
        ]
        if args.ko:
            cmd.append("--ko")
        cmd.extend(["--prompt_lang", args.prompt_lang])
        if args.option_ids4:
            cmd.extend(["--option_ids4", args.option_ids4])
        if args.option_ids5:
            cmd.extend(["--option_ids5", args.option_ids5])
        
        blue = "\033[94m"
        reset = "\033[0m"
        print(f"{blue}Model: {model}{reset}")
        print(f"{blue}Command: {' '.join(cmd)}{reset}")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"[FAIL] model={model} (returncode={proc.returncode})")
        else:
            print(f"[OK] model={model}")


if __name__ == "__main__":
    main()


