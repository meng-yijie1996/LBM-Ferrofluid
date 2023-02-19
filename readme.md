# LBM-Based Ferrofluid

<img src="demo/Ferrofluid.png" height="270">

### How to run the code

Buid this project

```bash
cd LBM-Ferrofluid
git submodule update --init --recursive
conda env create -f conda_env.yaml -n LBM_ferrofluid
conda activate LBM_ferrofluid
pip install -r requirements.txt
python setup.py build
```

Run demos

```bash
cd demo
python demo_3d_LBM_Rosensweig_instability.py
```

### Trouble Shooting

- If you have met a problem on windows that 'cl' cannot be found, go into your Visual Studio and find it in `Microsoft Visual Studio\20xx\Community\VC\Tools\MSVC\xx.xx.xxxxx\bin\Hostx64\x64`, then add them into your environment path.
