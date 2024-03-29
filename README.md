# NP-SemiSeg 

**NP-SemiSeg: When Neural Processes meet Semi-Supervised Semantic Segmentation**

Jianfeng Wang<sup>1</sup>, Daniela Massiceti<sup>2</sup>, Xiaolin Hu<sup>3</sup>, Vladimir Pavlovic<sup>4</sup> and   Thomas Lukasiewicz<sup>1</sup>

*University of Oxford*<sup>1</sup>, *Microsoft Research*<sup>2</sup>, *Tsinghua University*<sup>3</sup>,  *Rutgers University*<sup>4</sup> 

In [ICML 2023](https://proceedings.mlr.press/v202/wang23x.html)

Build
-----

please run with the following command:


```
conda env create -f NP-SemiSeg.yaml
conda activate NP-SemiSeg
```


Experiment
-----

We release the neural processes header (np_head.py) for semi-supervised semantic segmentation, and how it is used is shown in the two segmentation frameworks, namely U2PL and AugSeg.


Please download pretrained resnet50.pth and datasets at first. The datasets can be found in the original [U2PL](https://github.com/Haochen-Wang409/U2PL#readme) and [AugSeg](https://github.com/ZhenZHAO/AugSeg) repos. Here we provide download links of resnet50.pth for convenience. 

ResNet-50:

<table><tbody>
   <!-- START TABLE -->
   <!-- TABLE HEADER -->
   <th valign="bottom">Google Drive</th>
   <th valign="bottom">Baidu Disk</th>
   <!-- TABLE BODY -->
   <tr>
   <td align="center"><a href="https://drive.google.com/file/d/1O8eV7stVXcjbGyhemEI6Bk4ERiQr8iUN/view?usp=sharing">download</a></td>
   <td align="center"><a href="https://pan.baidu.com/s/1YNcVw-VOd2zlOZ-fDcfiDw">download</a>  (code: mhbx)  </td>
   </tr>
   </tbody></table> 


Please put the resnet50.pth in the "pretrained" directory and datasets in the "data" directory, in both U2PL and AugSeg. 

**For U2PL:**


```
cd experiments/cityscapes/744/np/

sh train.sh <num_gpu> <port>
```

After training, the model should be evaluated by

```
sh eval.sh
```

**For AugSeg:**

Please configure your yaml file in a running script "./scripts/run_abls_citys.sh", and then run:

```
sh ./scripts/run_abls_citys.sh
```



Citation
-----

```
@inproceedings{wang2023np,
  title={NP-SemiSeg: when neural processes meet semi-supervised semantic segmentation},
  author={Wang, Jianfeng and Massiceti, Daniela and Hu, Xiaolin and Pavlovic, Vladimir and Lukasiewicz, Thomas},
  booktitle={International Conference on Machine Learning},
  pages={36138--36156},
  year={2023},
  organization={PMLR}
}
```


Acknowledgement
-----

The released codes contain the contents from [U2PL](https://github.com/Haochen-Wang409/U2PL#readme) and [AugSeg](https://github.com/ZhenZHAO/AugSeg). We thank for their contributions. 
 
