
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
from pathlib import Path
from log import get_logger, set_logger
from data_transform import *


if __name__ == "__main__":
    # np.set_printoptions(threshold=np.inf, precision=2)
    np.set_printoptions(precision=2)
    logger = get_logger()
    log_dir = Path('log')
    set_logger(logger, log_dir / "consistency.log")

    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    datasources = []

    # commented
    datasource = dict()
    datasource['name'] = 'MNIST'
    datasource['image_size'] = 28
    datasource['n_channels'] = 1
    trainset = datasets.MNIST('../../datasets', download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))   # 6e4
    # testset = datasets.MNIST('../../datasets', download=True, train=False, transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))]))   # 1e4
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=4)
    test_loader_mile = DataLoader(trainset, batch_size=60000, shuffle=False, num_workers=4)
    datasource['train_loader'] = train_loader
    datasource['test_loader'] = test_loader
    datasource['test_loader_mile'] = test_loader_mile
    datasources.append(datasource)

    datasource = dict()
    datasource['name'] = 'CIFAR10'
    datasource['image_size'] = 32
    datasource['n_channels'] = 3
    trainset = datasets.CIFAR10(root='../../datasets', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))   # 6e4
    # testset = datasets.CIFAR10(root='../../datasets', train=False, download=True, transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))   # 1e4
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=4)
    test_loader_mile = DataLoader(trainset, batch_size=50000, shuffle=False, num_workers=4)
    datasource['train_loader'] = train_loader
    datasource['test_loader'] = test_loader
    datasource['test_loader_mile'] = test_loader_mile
    datasources.append(datasource)
    
    excel_dir = Path('doc')
    excel_dir.mkdir(exist_ok=True, parents=True)
    writer_mi = pd.ExcelWriter(excel_dir/'self_consistency_mi.xlsx', engine='xlsxwriter')
    writer_percentage = pd.ExcelWriter(excel_dir/'self_consistency_percentage.xlsx', engine='xlsxwriter')
    df_mi = pd.DataFrame([])
    df_percentage = pd.DataFrame([])

    for datasource in datasources:
        denominator = None
        baseline = Baseline(device, datasource)
        denominator, df_mi, df_percentage = baseline.export(denominator)
        df_mi.to_excel(writer_mi, sheet_name = datasource['name'] + '_' + baseline.__class__.__name__)
        df_percentage.to_excel(writer_percentage, sheet_name = datasource['name'] + '_' + baseline.__class__.__name__)
        logger.debug(f'df_mi {df_mi}\ndf_percentage {df_percentage}')

        data_processing = DataProcessing(device, datasource)
        _, df_mi, df_percentage = data_processing.export(denominator)
        df_mi.to_excel(writer_mi, sheet_name = datasource['name'] + '_' + data_processing.__class__.__name__)
        df_percentage.to_excel(writer_percentage, sheet_name = datasource['name'] + '_' + data_processing.__class__.__name__)
        logger.debug(f'df_mi {df_mi}\ndf_percentage {df_percentage}')

        additivity = Additivity(device, datasource)
        _, df_mi, df_percentage = additivity.export(denominator)
        df_mi.to_excel(writer_mi, sheet_name = datasource['name'] + '_' + additivity.__class__.__name__)
        df_percentage.to_excel(writer_percentage, sheet_name = datasource['name'] + '_' + additivity.__class__.__name__)
        logger.debug(f'df_mi {df_mi}\ndf_percentage {df_percentage}')

    writer_mi.close()
    writer_percentage.close()
    