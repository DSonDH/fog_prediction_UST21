# python ./code/step_2_train_hpo_ml.py -m station_code=SF_0006 stage=train pred_hour=1,3,6 model_name=cb, lgb, rf, xgb
# hydra optuna 에서는 pred_hour=1,3,6 model_name=cb, lgb, rf, xgb 이렇게 주면 
# serial한 조합으로 찾는게 아니라, 모두 다 tuning option이 되어 독립적인 search trial을 경험하지 못하게 됨

# 따라서 원시적이지만 독립적으로 for loop 돌려서 하기로 함
# python ./code/step_2_train_hpo_ml.py -m station_code=SF_0006 stage=train pred_hour=1,3,6 model_name=cb
# for station in SF_0002 SF_0003 SF_0004 SF_0005 SF_0006 SF_0007 SF_0008 SF_0009 SF_0010 SF_0011
for station in SF_0002 SF_0003 SF_0004 SF_0005 SF_0006 SF_0007 SF_0008 SF_0009 SF_0010 SF_0011
do
    for pred_hour in 1 3 6
    do
        for model in cb lgb rf xgb
        do
            if [ -d "/home/sdh/fog-generation-ml/data/log_exp5/2.1/$station/$pred_hour/$model" ]; then
                echo "$station/$pred_hour/$model already done"
            else
                echo
                echo
                echo "------------------------------------------------------------------------------------------------------------"
                echo "           Step: 2    Stage: Train    Station_Code: $station    Pred_Hour: $pred_hour    Model_Name: $model"
                echo "------------------------------------------------------------------------------------------------------------"
                export HYDRA_FULL_ERROR=1
                python ./code/step_2_train_hpo_ml.py -m stage=train station_code=$station pred_hour=$pred_hour model_cfg@_global_=$model "hydra.job.env_set.CUDA_VISIBLE_DEVICES='1'"
            fi
        done
    done
done


# for station in  SF_0001 SF_0002 SF_0003 SF_0004 SF_0005 SF_0006 SF_0007 SF_0008  SF_0009 SF_0010 SF_0011
# do
#     for pred_hour in 1 3 6
#     do
#         for model in cb lgb rf xgb
#         do
#             python ./code/step_3_find_best_HP_best_model.py stage=train station_code=$station pred_hour=$pred_hour model_name=$model
#         done
#     done
# done
