
from modules.utility import PM, bcolors

from modules.network.ModelModule import ModelFactory, ModelEnum
from modules.network.CriterionModule import CriterionFactory, CriterionEnum
from modules.network.OptimizerModule import OptimizerFactory, OptimizerEnum


class NetworkFactory():
    ModelEnum = ModelEnum
    CriterionEnum = CriterionEnum
    OptimizerEnum = OptimizerEnum

    def build(typeModel: ModelEnum, input_size_LSTM, hidden_size_LSTM, device,
              typeCriterion: CriterionEnum,
              typeOptimizer: OptimizerEnum):

        model = ModelFactory.build(typeModel, input_size_LSTM, hidden_size_LSTM, device)
        criterion = CriterionFactory.build(typeCriterion)
        optimizer = OptimizerFactory.build(typeOptimizer, model)

        PM.printI(bcolors.LIGHTRED+"Building model:"+bcolors.ENDC)
        PM.printI(bcolors.LIGHTYELLOW+"model: {}".format(typeModel)+bcolors.ENDC)
        PM.printI(bcolors.LIGHTYELLOW+"criterion: {}".format(typeCriterion)+bcolors.ENDC)
        PM.printI(bcolors.LIGHTYELLOW+"optimizer: {}".format(typeOptimizer)+bcolors.ENDC+"\n")

        # for parameter in model.parameters():
        #     PM.printD(str(parameter.size()))

        return model, criterion, optimizer

