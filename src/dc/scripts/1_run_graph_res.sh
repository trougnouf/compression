CURDIR=$(pwd)
METHODS=(graph_clicpro graph_last graph_denoisethencompress graph_add_dc graph_lownoise_2 graph_lownoise_3 graph_hq graph_85_onemodel graph_85_twomodels graph_lownoise graph_one)
for METHOD in ${METHODS[@]}; do
    python tools/graph_res.py --config "config/${METHOD}.yaml"
    cd "$CURDIR/../../results/dc/${METHOD}.yaml"
    rm *-crop.pdf
    for FILE in *.pdf; do
        pdfcrop "${FILE}"
    done
    rm *-crop-crop*
    cd $CURDIR
done
python tools/graph_res.py
cd "$CURDIR/../../results/dc/"
for FILE in *.pdf; do
  pdfcrop "${FILE}"
done
rm *-crop-crop*
cd $CURDIR
# for METHOD in ${METHODS[@]}; do
#     # python tools/graph_res.py --config "config/${METHOD}.yaml"
#     cd "$CURDIR/../../results/dc/${METHOD}.yaml"
#     rm *-crop.pdf
#     for FILE in *.pdf; do
#         pdfcrop "${FILE}"
#     done
# done
# # python tools/graph_res.py
# cd "$CURDIR/../../results/dc/"
# for FILE in *.pdf; do
#   pdfcrop "${FILE}"
# done

#python tools/graph_res.py --config config/graph_clicpro.yaml
#python tools/graph_res.py --config config/graph_last.yaml
#python tools/graph_res.py --config config/graph_denoisethencompress.yaml
#python tools/graph_res.py --config config/graph_add_dc.yaml
#python tools/graph_res.py --config config/graph_lownoise_2.yaml
#python tools/graph_res.py --config config/graph_lownoise_3.yaml
#python tools/graph_res.py --config config/graph_85_onemodel.yaml && python tools/graph_res.py --config config/graph_85_twomodels.yaml && python tools/graph_res.py --config config/graph_lownoise.yaml && python tools/graph_res.py --config config/graph_one.yaml && python tools/graph_res.py


#DIRS=
#for FILE in ./*.pdf; do
#  pdfcrop "${FILE}"
#done
