python train.py --save_path ./ckpt/DP_CLIP 
declare -a dataset=(MVTec MPDD BTAD Colon_clinicDB Colon_colonDB Colon_Kvasir Colon_cvc300 Liver Brain Retina)
save_path="./ckpt/DP_CLIP"
for i in "${dataset[@]}"; do
    python test.py --save_path $save_path --dataset $i 
done
