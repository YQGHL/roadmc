#!/usr/bin/env bash
# 重新生成 README 的两张管线图：TikZ 源 -> xelatex -> PDF -> 300 dpi PNG
# 与论文插图同一套风格（standalone + Liberation Sans + 灰细线框 + 彩色 12% 填充）。
#
# 用法：bash readmeimage/build_diagrams.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/tex"

for name in synthesis_pipeline training_pipeline; do
  xelatex -interaction=nonstopmode -halt-on-error "$name.tex" > /dev/null
  pdftoppm -r 300 -png -singlefile "$name.pdf" "$HERE/$name"
  echo "built $HERE/$name.png"
done

# 清掉中间文件，只留 .tex / .pdf / .png
rm -f ./*.aux ./*.log
echo "done"
