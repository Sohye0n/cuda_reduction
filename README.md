# cuda_reduction
optimizing cuda addition kernel
reduction을 이해하기 위해 7단계에 걸쳐 커널을 최적화해보았습니다.

</br></br>

## 🗂️ 파일 구조
```
main.cpp
│
├── utils.h
│   └── utils.cpp
│
└── addition.h
    └── addition.cu
```
</br>

## ⚙️ 개발 환경
- `CUDA 12.6`
</br>

## 🛠️ 설치 및 실행 방법
- 빌드 명령어 : `make all`
- 실행 방법 &nbsp; &nbsp;: `./main -v 3 -n 1000`
</br>

## 📌 구현 단계
#### ver1
- interleaved addressing
- 인접한 데이터에서 시작해 점차 멀리 떨어진 데이터에 접근하는 방식.
#### ver2
- interleaved addressing
- 인접한 데이터에서 시작해 점차 멀리 떨어진 데이터에 접근하는 방식.
- ver1에서의 divergence 문제를 해결함.
#### ver3
- sequential addressing
- 떨어진 데이터에서 시작해 점차 인접한 데이터에 접근하는 방식.
- ver2에서의 bank conflict를 해결함.
#### ver4
- 처음 공유 메모리에 로드하는 과정에서 미리 두개의 원소를 합한다.
#### ver5
- loop unrolling 1
#### ver6
- loop unrolling 2
#### ver7
- algorithmic cascading
- 스레드 개수를 줄이고, 한 스레드의 연산량을 더 늘렸다.
</br>

## ✍️ 구현 원리
이 링크</a>에 업데이트 중입니다.
