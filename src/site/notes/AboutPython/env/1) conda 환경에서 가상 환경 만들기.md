---
{"dg-publish":true,"permalink":"/AboutPython/env/1) conda 환경에서 가상 환경 만들기/","tags":["python"],"noteIcon":""}
---

## 온라인 환경 

1. 먼저 아나콘다 프롬프트를 실행한다. 
![](https://i.imgur.com/gVeo07n.png)
- 처음에는 base 상태인 것을 볼 수 있다. 

2.  여기서 아래와 같이 입력해준다 
- -n 뒤의 python312 는 가상환경의 이름을 / 그뒤의 python=3.12 는 파이썬의 버젼을 의미한다. 
  여기서 띄어쓰기 유의해서 입력해야 에러가 뜨지 않음.
  
```cmd
conda create -n python312 python=3.12
```

- active 환경 deactive 환경의 명령어는 아래와 같다 
```cmd 
conda activate python312 #active 
conda deactivate #deactive 
```


## 오프라인 환경

찾아보니 오프라인으로 가상환경을 생성하는 방법은 없는듯 하다. ~~(그래서 그렇게 에러를 냈구나..!)

1. 아나콘다 프롬프트를 시작한다.
![](https://i.imgur.com/gVeo07n.png)

```
conda create -n <새로운 가상환경명> --clone <복제할 가상환경명> 

```
- 위와같이 가상환경 복제를 통해 만들어야한다..!! 

### 아나콘다 오프라인으로 가상환경 추출 및 설치 
conda env list : 가상환경명, 
conda list : 해당 가상환경에 패키지 목록 및 버전. 

- conda pack 설치 후 가상환경 파일로 만들기 
```cmd 
conda install –c conda-forge conda-pack 
``` 
`conda pack -n <추출할 가상환경명>  -o <추출할 가상환경 압축파일.tar.gz>`

![](https://i.imgur.com/2t5zsaU.png)


- 오프라인 환경의 경우에는 필요한 파이썬 파일과 패키지셋을 전부 다운로드 해야한다. 