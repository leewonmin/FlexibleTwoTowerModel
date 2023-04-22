## Dataset

### We use the CiteULike dataset files provided by the impolementation of the 'Shared Neural Item Representations for Completely Cold Start Problem': https://github.com/rakutentech/shared_rep.
Check out the trainset.npy, item_dict.pkl, user_dict.pkl, test_items_ids.pkl, val_grtr.pkl files here.

We provide additional dataset file for our model in the citeulike folder, and the MLIMDb dataset can be found in the mlimdb folder.

## Example to run the codes
### git main.py --ds citeulike --model flexible_model --remove_item --epoch 50 --batch_size32 --lr 0.001
