import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from model import HAN
from utils import load_data_HAN, accuracy, knn_classifier

from sklearn.metrics import precision_score, recall_score, f1_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=88,
                        help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Number of hidden dimension.')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads for node-level attention.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Alpha for the leaky_relu.')
    parser.add_argument('--q_vector', type=int, default=256,
                        help='The dimension for the semantic attention embedding.')
    parser.add_argument('--patience', type=int, default=100,
                        help='Number of epochs with no improvement to wait before stopping')
    parser.add_argument('--dataset', type=str, default='DBLP',
                        help='The dataset to use for the model.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load the data
    # features: (N, F), meta_path_list: (M, N, N), labels: (N, 1)
    features, meta_path_list, labels, idx_train, idx_val, idx_test = load_data_HAN(args.dataset)
    model = HAN(feature_dim=features.shape[1],
                hidden_dim=args.hidden_dim,
                num_classes=int(labels.max()) + 1,
                dropout=args.dropout,
                num_heads=args.num_heads,
                alpha=args.alpha,
                q_vector=args.q_vector)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        meta_path_list = [meta_path.cuda() for meta_path in meta_path_list]

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    best_loss_val = float('inf')
    patience = args.patience
    counter = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=20, 
        verbose=True
    )

    for epoch in range(args.epochs):
        model.train()
        output = model(features, meta_path_list)
        loss_train = criterion(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            output = model(features, meta_path_list)
            loss_val = criterion(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            
            scheduler.step(loss_val)

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train),
                  'acc_train: {:.4f}'.format(acc_train),
                  'loss_val: {:.4f}'.format(loss_val),
                  'acc_val: {:.4f}'.format(acc_val))

            if loss_val < best_loss_val:
                best_loss_val = loss_val
                # Save the best model state if needed
                torch.save(model.state_dict(), 'best_model.pth')
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    # Load the best model state
    model.load_state_dict(torch.load('best_model.pth'))

    model.eval()
    output = model(features, meta_path_list)
    X = output[idx_test].detach().cpu().numpy()
    y = labels[idx_test].detach().cpu().numpy()
    knn_classifier(X, y, seed=args.seed)
    preds = output[idx_test].argmax(dim=1).cpu().numpy()
    
    # Macro scores
    precision_macro = precision_score(y, preds, average='macro')
    recall_macro = recall_score(y, preds, average='macro')
    f1_macro = f1_score(y, preds, average='macro')

    # Micro scores
    precision_micro = precision_score(y, preds, average='micro')
    recall_micro = recall_score(y, preds, average='micro')
    f1_micro = f1_score(y, preds, average='micro')

    print(f"\nTest Metrics:")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro):    {recall_macro:.4f}")
    print(f"F1 Score (macro):  {f1_macro:.4f}")
    print(f"Precision (micro): {precision_micro:.4f}")
    print(f"Recall (micro):    {recall_micro:.4f}")
    print(f"F1 Score (micro):  {f1_micro:.4f}")
