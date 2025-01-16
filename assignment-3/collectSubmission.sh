#!/bin/bash
set -euo pipefail

#### Uncomment the following if using Linux. Change accordingly for other OS #################
#sudo apt-get update
#sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-plain-generic


pip install PyPDF2

CODE=(
	"cs682/transformer_layers.py"
	"cs682/simclr/contrastive_loss.py"
	"cs682/simclr/data_utils.py"
	"cs682/simclr/utils.py"
)
NOTEBOOKS=(
	"Transformer_Captioning.ipynb"
	"Self_Supervised_Learning.ipynb"
	"StyleTransfer.ipynb"
)
PDFS=(
	"Transformer_Captioning.ipynb"
	"Self_Supervised_Learning.ipynb"
	"StyleTransfer.ipynb"
)

FILES=( "${CODE[@]}" "${NOTEBOOKS[@]}" )
ZIP_FILENAME="a3_code_submission.zip"
PDF_FILENAME="a3_inline_submission.pdf"

for FILE in "${FILES[@]}"
do
	if [ ! -f ${FILE} ]; then
		echo -e "${C_R}Required file ${FILE} not found, Exiting.${C_E}"
		exit 0
	fi
done

echo -e "### Zipping file ###"
rm -f ${ZIP_FILENAME}
zip -q "${ZIP_FILENAME}" -r ${NOTEBOOKS[@]} $(find . -name "*.py") $(find . -name "*.pyx") -x "makepdf.py"


echo -e "### Done! Please submit ${ZIP_FILENAME} and ${PDF_FILENAME} to Gradescope. ###"
