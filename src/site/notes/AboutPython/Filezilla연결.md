---
{"dg-publish":true,"permalink":"/AboutPython/Filezilla연결/","title":"FTP 파이썬 컨트롤","tags":["python","FTP","File"],"noteIcon":""}
---


<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>
많은 회사에서 FTP 연결프로그램으로 FileZilla를 사용 하고 있다. 이를 파이썬으로 연결하여 파일 업로드 및 다운로드를 하는 함수를 만들어 보았다. (향후 자동화에 해당 함수를 쓰기위함)


```python
import paramiko
import os


host = 'APKRP-WFSR193'
port = 22
user = 'username'
password = 'password'

transport = paramiko.Transport((host,port))
transport.connect(username=user, password = password)
sftp = paramiko.SFTPClient.from_transport(transport)


 

def getftp(remote_path,local_path):
  # sftp = paramiko.SFTPClient.from_transport(transport)
  sftp.chdir('/')
  sftp.get(remote_path,local_path)
  # sftp.close()
  # transport.close()


 

def toftp(local_path,remote_path):
  # sftp = paramiko.SFTPClient.from_transport(transport)
  sftp.chdir('/')
  sftp.remove(remote_path)
  sftp.put(local_path,remote_path)

  # sftp.close()

  # transport.close()

def file_list(remote_path):
  sftp.chdir('/')
  return sftp.listdir(remote_path)

 

# getftp('AI_LNC_Summary_20230216.xlsx', r'C:\Users\gukim00\Desktop\DB추출\AI_LNC_Summary_20230216.xlsx')

# toftp( r'C:\Users\gukim00\Desktop\DB추출\PP2_2.xlsx','/path/PP2_2.xlsx')
```
