# Python-boilerplate
> A brief description of your project, what it is used for and how does life get awesome when someone starts to use it.

This repository is boilerplate for golang.
Please find the parts that say 'FIXED_ME' and use them after modifying them.

## Features
> What's all the bells and whistles this project can perform?
* What's the main functionality
* You can also do another thing
* If you get really randy, you can even do this


## Installing / Getting started
> A quick introduction of the minimal setup you need to get a hello world up & running.
...

## Initial Configuration
> A quick introduction of the minimal setup you need to get a hello world up &
running.

### Setup for development mode
```bash
make setup
```

### Run application in development mode

```bash
# default port is `8088`
make dev
```

### Lint
```bash
make lint
```

### Formatting
```bash
make format
```

### Testing
```bash
make test
```

## Documentation
- [Envionment](./doc/VARIABLES.md)

## Folder structure

```bash
.
├── Makefile # 환경 설정, 마이그레이션, 개발 등 필요로 하는 스크립트를 packaging 하기 위해서
├── README.md
├── app
│   ├── api # presentation logic
│   │   ├── graphql # 예를 들기 위해서 해 둔 것임..
│   │   ├── grpc # grpc routring 정보들이 들어 감
│   │   └── http # rest api 정보 들이 들어감 http vs rest 고민중
│   ├── config # 설정 파일
│   └── main.py # application main file
├── deploy # 배포하기 위해서 필요한 파일들이 들어감
│   ├── Dockerfile # Dockerfile
│   └── Dockerfile.migration # Migration을 위한 Dockerfile
├── development # 개발시 세팅하거나 필요한 것들을 넣어둠
│   └── docker-compose.yml
├── doc # 프로젝트 관련된 문서들이 저장되어 있는 것
│   ├── VARIABLES.md
│   └── asset # 문서에 필요한 파일(그림 등등)
├── pyproject.toml
├── resource # 프로젝트를 수행하기 위해서 필요로하는 py 파일 이외에 파일들을 저장 하기 위한 directory
├── test # test를 모아두는 directory
└── script # 개발 및 배포 등등 프로젝트 로직과 관련 없는 script 모아두는 폴더
```

## Reference
> link for related documents.
