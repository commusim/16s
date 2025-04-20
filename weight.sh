
python dl.py --mode regression --method Hybrid --weight_decay 1e-4
python dl.py --mode regression --method Hybrid --weight_decay 1e-3
python dl.py --mode regression --method Hybrid --weight_decay 1e-2
python dl.py --mode regression --method Hybrid --weight_decay 1e-1

python dl.py --mode regression --method MLP --weight_decay 1e-4
python dl.py --mode regression --method MLP --weight_decay 1e-3
python dl.py --mode regression --method MLP --weight_decay 1e-2
python dl.py --mode regression --method MLP --weight_decay 1e-1


python dl.py --mode classification --method Hybrid --weight_decay 1e-4
python dl.py --mode classification --method Hybrid --weight_decay 1e-3
python dl.py --mode classification --method Hybrid --weight_decay 1e-2
python dl.py --mode classification --method Hybrid --weight_decay 1e-1

python dl.py --mode classification --method MLP --weight_decay 1e-4
python dl.py --mode classification --method MLP --weight_decay 1e-3
python dl.py --mode classification --method MLP --weight_decay 1e-2
python dl.py --mode classification --method MLP --weight_decay 1e-1