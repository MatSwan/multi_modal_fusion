from models.vgg_16_model.fusion_vgg16 import FusionVGG16
from options.enum_options import TrainData, FusionOptions, TrainingOptions


def experimental_name_parser(fusion_name: str, dataset_name: str, base_experiment_name='vggfusion') -> str:
    return base_experiment_name + '_' + fusion_name + dataset_name


if __name__ == "__main__":
    train_opt: TrainingOptions = TrainingOptions().parse_args()
    fusion_opt: FusionOptions = FusionOptions.FULLYCONNECTED
    experimental_data = TrainData.INTEL
    experiment_name = experimental_name_parser(fusion_opt.name, experimental_data.name)

    fusion_experiment = FusionVGG16(root=train_opt.root, experiment_name=experiment_name, train_data=experimental_data,
                                    test_data=experimental_data.get_test_data(), device=train_opt.device,
                                    epoch_count=train_opt.epoch_count, pretrain=fusion_opt.pretrain,
                                    pretrain_epoch=fusion_opt.pretrain_epoch, batch_size=train_opt.batch_size,
                                    fusion=fusion_opt.value,
                                    number_of_modalities=experimental_data.number_of_modalities,
                                    number_of_classes=experimental_data.number_of_classes,
                                    image_channel=experimental_data.image_channel,
                                    image_height=experimental_data.image_height,
                                    image_width=experimental_data.image_width, number_of_runs=train_opt.number_of_runs
                                    )
    fusion_experiment.run()

####Stuff to DO #####:
# todo fix fusion options
# todo create a FusionFactory class with a "__call__(int, int) -> nn.Module" signature
# todo
