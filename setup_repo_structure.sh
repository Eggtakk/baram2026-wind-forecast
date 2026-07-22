#!/usr/bin/env bash
# baram2026-wind-forecast 디렉토리 구조 일괄 생성 스크립트
#
# 사용법:
#   1) GitHub에서 만든 레포를 클론 후 그 폴더 안으로 이동
#      git clone https://github.com/<owner>/baram2026-wind-forecast.git
#      cd baram2026-wind-forecast
#   2) 이 스크립트를 레포 루트에 저장 후 실행 권한 부여
#      chmod +x setup_repo_structure.sh
#   3) 실행
#      ./setup_repo_structure.sh
#
# 이미 있는 파일(README.md, .gitignore 등)은 덮어쓰지 않습니다.

set -e

echo "📁 디렉토리 생성 중..."

mkdir -p configs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p src/data
mkdir -p src/features
mkdir -p src/metrics
mkdir -p src/models/group1
mkdir -p src/models/group2
mkdir -p src/models/group3
mkdir -p notebooks/eda
mkdir -p notebooks/experiments
mkdir -p experiments
mkdir -p submissions
mkdir -p docs/presentation
mkdir -p scripts

echo "📄 파일 생성 중..."

# --- configs ---
touch configs/group1.yaml
touch configs/group2.yaml
touch configs/group3.yaml

# --- src/data ---
[ -f src/data/load.py ] || cat > src/data/load.py << 'EOF'
"""원본 CSV(ldaps, gfs, labels, scada) 로딩 공통 함수."""
EOF

[ -f src/data/grid_match.py ] || cat > src/data/grid_match.py << 'EOF'
"""info.xlsx 좌표 기반 터빈-기상격자 매칭 로직."""
EOF

[ -f src/data/merge.py ] || cat > src/data/merge.py << 'EOF'
"""LDAPS/GFS/label 병합 및 train/test 정렬."""
EOF

# --- src/features ---
[ -f src/features/build_features.py ] || cat > src/features/build_features.py << 'EOF'
"""파생 변수(고도별 풍속, lag/rolling 등) 생성."""
EOF

# --- src/metrics ---
[ -f src/metrics/evaluation.py ] || cat > src/metrics/evaluation.py << 'EOF'
"""1-NMAE, FICR 스코어 재현 함수 (팀 공용)."""
EOF

# --- src (train/inference 분리 - 제출 규정 대응) ---
[ -f src/train.py ] || cat > src/train.py << 'EOF'
"""학습 진입점. 추론 코드(inference.py)와 반드시 분리 유지."""
EOF

[ -f src/inference.py ] || cat > src/inference.py << 'EOF'
"""추론 진입점. Private Score 복원 가능해야 함."""
EOF

# --- experiments ---
[ -f experiments/logs.md ] || cat > experiments/logs.md << 'EOF'
# 실험 로그

| 날짜 | 담당자 | 그룹 | 모델/config | 1-NMAE | FICR | 리더보드 점수 | 비고 |
|---|---|---|---|---|---|---|---|
EOF

# --- scripts ---
[ -f scripts/make_submission.py ] || cat > scripts/make_submission.py << 'EOF'
"""sample_submission.csv 형식에 맞춰 최종 제출 CSV 생성."""
EOF

# --- docs ---
[ -f docs/data_description.md ] || touch docs/data_description.md
[ -f docs/evaluation_note.md ] || touch docs/evaluation_note.md

# --- 빈 디렉토리 유지용 .gitkeep (git은 빈 폴더를 추적하지 않음) ---
for d in data/raw data/processed src/models/group1 src/models/group2 src/models/group3 \
         notebooks/eda notebooks/experiments submissions docs/presentation; do
  [ -n "$(ls -A "$d" 2>/dev/null)" ] || touch "$d/.gitkeep"
done

# --- environment.yml (없을 때만 생성) ---
[ -f environment.yml ] || cat > environment.yml << 'EOF'
name: baram2026
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pandas
  - numpy
  - scikit-learn
  - pip
  - pip:
      - lightgbm
      - xgboost
EOF

# --- .gitignore 에 데이터/모델 관련 항목 추가 (기존 내용 보존, 중복 방지) ---
GITIGNORE_ADD="data/
submissions/*.csv
*.ckpt
*.pkl
*.pt
wandb/
.env"

touch .gitignore
for line in "data/" "submissions/*.csv" "*.ckpt" "*.pkl" "*.pt" "wandb/" ".env"; do
  grep -qxF "$line" .gitignore || echo "$line" >> .gitignore
done

echo "✅ 완료! 아래 명령으로 확인하세요:"
echo "   find . -not -path '*/.git*' | sort"
