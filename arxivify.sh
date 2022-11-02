#!/bin/bash

if [[ -d arxiv ]]; then
  rm -r arxiv
fi
mkdir arxiv

cp -r figures dlde_neurips_2022.sty main.bib main.bbl main.tex terms.tex minted.sty arxiv

# Workaround for minted to forgo -shell-escape
cp -r _minted-main arxiv
sed -i 's/finalizecache/frozencache/' arxiv/main.tex

tar -czvf arxiv.tar.gz arxiv
