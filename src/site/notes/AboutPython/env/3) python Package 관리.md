---
{"dg-publish":true,"permalink":"/AboutPython/env/3) python Package 관리/","tags":["python"],"noteIcon":""}
---


오프라인 패키지 설치 
- conda 환경의 경우
- cd 설치파일 경로를 통해 설치파일이 있는 경로로 이동한다. 
```cmd 
conda install --offline pandas-2.1.2.tar.gz
```

폴더에 있는 모든 패키지 다운로드 
```cmd 
pip install --no-index --find-links="./" -r .\requirements.txt
```

모든 패키지를 받을 필요가없는 경우, 다음 명령어로 패키지를 하나씩 설치 
```cmd 
pip install --find-links="./" [패키지 이름] 
```





