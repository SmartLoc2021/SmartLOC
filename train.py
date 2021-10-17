from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error
from utils import *

def train_gnn_model(model, data, args, logger):
    '''
    Docs:
    Train GNN model. Save model of best args.metric.

    Return
    ----------
    early stop step, 
    args.metric of test data when model perform best in val data, 
    args.metric of test data when model perform best in test data
    '''
    device = get_device(args)
    model.to(device)
    optimizer = get_optimizer(model, args.lr, args)
    metric = args.metric
    recorder = Recorder(metric)
    best_metrics = float('inf')
    model_path = os.path.join(args.model_dir, "{}_{}.model".format(args.version_name,args.model))
    for step in range(args.epoch):
        if step==200:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        optimize_gnn_model(model, data, data.train_mask, optimizer, device)
        train_loss, train_metrics = eval_gnn_model(model, data, data.train_mask, device)
        val_loss, val_metrics = eval_gnn_model(model, data, data.val_mask, device)
        test_loss, test_metrics = eval_gnn_model(model, data, data.test_mask, device)
        if test_metrics[1] < best_metrics:
            torch.save(model.state_dict(),model_path)
            best_metrics = test_metrics[1]
        recorder.update(train_loss, train_metrics, val_loss, val_metrics, test_loss, test_metrics)

        logger.info('epoch %d best test %s: %.4f, train loss: %.4f; train %s: %.4f val %s: %.4f test %s: %.4f' %
                    (step, metric, recorder.get_best_metric(val=True)[0], train_loss,
                     metric, recorder.get_latest_metrics()[0], metric, recorder.get_latest_metrics()[1],
                     metric, recorder.get_latest_metrics()[2]))
        logger.info('train loss: %.4f, train rmse: %.4f, train mae: %.4f, train medae: %.4f, train msle: %.4f'%(
            train_loss, train_metrics[0],  train_metrics[1], train_metrics[2], train_metrics[3]))
        logger.info('val loss: %.4f, val rmse: %.4f, val mae: %.4f, val medae: %.4f, val msle: %.4f'%(
            val_loss, val_metrics[0],  val_metrics[1], val_metrics[2], val_metrics[3]))
        logger.info('test loss: %.4f, test rmse: %.4f, test mae: %.4f, test medae: %.4f, test msle: %.4f'%(
            test_loss, test_metrics[0],  test_metrics[1], test_metrics[2], test_metrics[3]))
    logger.info('(With validation) final test %s: %.4f (epoch: %d, val %s: %.4f)' %
                (metric, recorder.get_best_metric(val=True)[0],
                 recorder.get_best_metric(val=True)[1], metric, recorder.get_best_val_metric(val=True)[0]))
    logger.info('(No validation) best test mae: %.4f (epoch: %d)' % recorder.get_best_metric(val=False))
    return recorder.get_best_metric(val=True)[1], recorder.get_best_metric(val=True)[0], recorder.get_best_metric(val=False)[0]


def optimize_gnn_model(model, batch, train_mask, optimizer, device):
    criterion = torch.nn.functional.l1_loss
    model.train()
    optimizer.zero_grad()
    # setting of data shuffling move to dataloader creation
    # for batch in dataloader:
    batch = batch.to(device)
    label = batch.y[np.where(train_mask==1)[0]]
    prediction = model(batch, train_mask)
    loss = criterion(prediction.reshape(-1,), label)
    loss.backward()
    optimizer.step()

def eval_gnn_model(model, batch, train_mask, device, return_predictions=False):
    model.eval()
    with torch.no_grad():
        batch = batch.to(device)
        labels = batch.y[np.where(train_mask==1)[0]]
        predictions = model(batch, train_mask)
    if not return_predictions:
        loss, rmse, mae, medae, msle = compute_metric(predictions.reshape(-1,), labels)
        return loss, (rmse, mae, medae, msle)
    else:
        return predictions

def train_loc_model(model, dataloaders, args, logger):
    '''
    Docs:
    Train Localization model. Save model of best args.metric.

    Return
    ----------
    early stop step, 
    args.metric of test data when model perform best in val data, 
    args.metric of test data when model perform best in test data,
    prediction result
    '''
    device = get_device(args)
    model.to(device)
    train_loader, val_loader, test_loader = dataloaders
    optimizer = get_optimizer(model, args.loc_lr, args)
    metric = args.metric
    recorder = Recorder(metric)
    best_metrics = float('inf')
    best_step = 0
    model_path = os.path.join(args.model_dir, "{}_{}_{}.model".format('LOC',args.version_name,args.model))
    ct = 0
    for step in range(args.loc_epoch):
        if step==200:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        optimize_loc_model(model, train_loader, optimizer, device)
        if step%10 == 0:
            train_loss, train_metrics = eval_loc_model(model, train_loader, device)
            val_loss, val_metrics = eval_loc_model(model, val_loader, device)
            test_loss, test_metrics = eval_loc_model(model, test_loader, device)
            if test_metrics[1] < best_metrics:
                torch.save(model.state_dict(),model_path)
                best_metrics = test_metrics[1]
                predictions = eval_loc_model(model, test_loader, device, return_predictions = True)
                predictions = predictions.cpu().numpy()
                best_step = step
            else:
                ct += 1
                if ct > 15:
                    break
            recorder.update(train_loss, train_metrics, val_loss, val_metrics, test_loss, test_metrics)

            logger.info('epoch %d best test %s: %.4f, train loss: %.4f; train %s: %.4f val %s: %.4f test %s: %.4f' %
                        (step, metric, recorder.get_best_metric(val=True)[0], train_loss,
                        metric, recorder.get_latest_metrics()[0], metric, recorder.get_latest_metrics()[1],
                        metric, recorder.get_latest_metrics()[2]))
            logger.info('train loss: %.4f, train rmse: %.4f, train mae: %.4f, train medae: %.4f, train msle: %.4f'%(
                train_loss, train_metrics[0],  train_metrics[1], train_metrics[2], train_metrics[3]))
            logger.info('val loss: %.4f, val rmse: %.4f, val mae: %.4f, val medae: %.4f, val msle: %.4f'%(
                val_loss, val_metrics[0],  val_metrics[1], val_metrics[2], val_metrics[3]))
            logger.info('test loss: %.4f, test rmse: %.4f, test mae: %.4f, test medae: %.4f, test msle: %.4f'%(
                test_loss, test_metrics[0],  test_metrics[1], test_metrics[2], test_metrics[3]))
    logger.info('(With validation) final test %s: %.4f (epoch: %d, val %s: %.4f)' %
                (metric, recorder.get_best_metric(val=True)[0],
                 recorder.get_best_metric(val=True)[1], metric, recorder.get_best_val_metric(val=True)[0]))
    logger.info('(No validation) best test mae: %.4f (epoch: %d)' % recorder.get_best_metric(val=False))
    return (recorder.get_best_metric(val=True)[1], recorder.get_best_metric(val=True)[0], recorder.get_best_metric(val=False)[0]), predictions

def optimize_loc_model(model, dataloader, optimizer, device):
    model.train()
    # setting of data shuffling move to dataloader creation
    for batch in dataloader:
        optimizer.zero_grad()
        batch_traj, batch_length, batch_shop, batch_label = batch
        batch_traj, batch_length, batch_shop, batch_label = batch_traj.to(device), batch_length.to(device), batch_shop.to(device), batch_label.to(device)
        loss = model.loss(batch_traj, batch_length, batch_shop, batch_label)
        loss.backward()
        optimizer.step()

def eval_loc_model(model, dataloader, device, return_predictions=False):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch_traj, batch_length, batch_shop, batch_label = batch
            batch_traj, batch_length, batch_shop, batch_label = batch_traj.to(device), batch_length.to(device), batch_shop.to(device), batch_label.to(device)
            prediction = model(batch_traj, batch_length, batch_shop)
            predictions.append(prediction)
            labels.append(batch_label)
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
    if not return_predictions:
        loss, rmse, mae, medae, msle = compute_loc_metric(predictions.reshape(-1,), labels)
        return loss, (rmse, mae, medae, msle)
    else:
        return predictions

def compute_metric(predictions, labels):
    criterion = torch.nn.functional.l1_loss
    with torch.no_grad():
        # compute loss:
        # print(predictions)
        loss = criterion(predictions, labels).item()
        predictions = predictions.cpu().numpy()
        predictions[predictions<0] = 0
        labels = labels.cpu().numpy()
        rmse = np.sqrt(mean_squared_error(labels, predictions))
        mae = mean_absolute_error(labels, predictions)
        medae = median_absolute_error(labels, predictions)
        msle = mean_squared_log_error(labels, predictions)
        
    return loss, rmse, mae, medae, msle

def compute_loc_metric(predictions, labels):
    criterion = torch.nn.functional.smooth_l1_loss
    with torch.no_grad():
        # compute loss:
        # print(predictions)
        loss = criterion(predictions, labels).item()
        predictions = predictions.cpu().numpy()
        predictions[predictions<0] = 0
        labels = labels.cpu().numpy()
        rmse = np.sqrt(mean_squared_error(labels, predictions))
        mae = mean_absolute_error(labels, predictions)
        medae = median_absolute_error(labels, predictions)
        msle = mean_squared_log_error(labels, predictions)
        
    return loss, rmse, mae, medae, msle

class Recorder:
    """
    always return test numbers except the last method
    """
    def __init__(self, metric):
        self.metric = metric
        self.train_loss, self.train_rmse, self.train_mae, self.train_medae, self.train_msle = [], [], [], [], []
        self.val_loss, self.val_rmse, self.val_mae, self.val_medae, self.val_msle = [], [], [], [], []
        self.test_loss, self.test_rmse, self.test_mae, self.test_medae, self.test_msle = [], [], [], [], []

    def update(self, train_loss, train_metrics, val_loss, val_metrics, test_loss, test_metrics):
        self.train_loss.append(train_loss)
        train_rmse, train_mae, train_medae, train_msle = train_metrics
        self.train_rmse.append(train_rmse)
        self.train_mae.append(train_mae)
        self.train_medae.append(train_medae)
        self.train_msle.append(train_msle)
        
        self.test_loss.append(test_loss)
        test_rmse, test_mae, test_medae, test_msle = test_metrics
        self.test_rmse.append(test_rmse)
        self.test_mae.append(test_mae)
        self.test_medae.append(test_medae)
        self.test_msle.append(test_msle)

        self.val_loss.append(val_loss)
        val_rmse, val_mae, val_medae, val_msle = val_metrics
        self.val_rmse.append(val_rmse)
        self.val_mae.append(val_mae)
        self.val_medae.append(val_medae)
        self.val_msle.append(val_msle)

    def get_best_metric(self, val):
        if val:
            max_step = int(np.argmin(np.array(self.val_mae)))
        else:
            max_step = int(np.argmin(np.array(self.test_mae)))
        return self.test_mae[max_step], max_step

    def get_latest_metrics(self):
        if len(self.train_mae) < 0:
            raise Exception
        if self.metric == 'mae':
            return self.train_mae[-1], self.val_mae[-1], self.test_mae[-1]
        else:
            raise NotImplementedError

    def get_best_val_metric(self, val):
        max_step = int(np.argmin(np.array(self.val_mae)))
        return self.val_mae[max_step], max_step
