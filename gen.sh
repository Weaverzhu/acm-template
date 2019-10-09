pandoc --template algo --filter pandoc/minted.py --pdf-engine=xelatex --no-highlight --pdf-engine-opt="-shell-escape" -o template.tex --from markdown -V mainfont="PingFang SC" -V monofont="Monaco" -V sansfont="PingFang SC" -V CJKmainfont="PingFang SC" -V secnumdepth=2 -V --number-sections --toc -V include-before="\renewcommand\labelitemi{$\bullet$}" -V header-includes="\usepackage{minted}" -V geometry="margin=2cm" *-*.md
latexmk -xelatex -shell-escape template.tex
latexmk -c
