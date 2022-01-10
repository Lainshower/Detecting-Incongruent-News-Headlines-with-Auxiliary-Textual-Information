import argparse
import tensorflow as tf

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, type=str, default='test')
parser.add_argument('--max_tit', type=int, default=29)
parser.add_argument('--max_sub', type=int, default=114)
parser.add_argument('--max_body', type=int, default=35)
parser.add_argument('--max_sent', type=int, default=12)
parser.add_argument('--max_cap', type=int, default=24)
parser.add_argument('--drop', type=float, default=0.1)
parser.add_argument('--hidden1', type=int, default=512)
parser.add_argument('--hidden2', type=int, default=256)
parser.add_argument('--batch', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=15)
args = parser.parse_args()

if args.mode == 'train':
	from train import Trainer
	trainer = Trainer(max_tit=args.max_tit, max_sub=args.max_sub, max_body=args.max_body, max_sent=args.max_sent, max_cap=args.max_cap,
	drop=args.drop, hidden1=args.hidden1, hidden2=args.hidden2, batch=args.batch, epochs=args.epochs)
	trainer.training()

elif args.mode =='test':
	print("!!!!!!!!")
	from testing import test
	test()

else:
	raise ValueError ("please check --mode argument")