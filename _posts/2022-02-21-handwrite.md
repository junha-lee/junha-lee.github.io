---
title: Deep Reinforcement learning
date: 2020-12-23
category: RL
tags:
    - RL
    - Project
toc: true
author_profile: false
sidebar:
  nav: "docs"
---

## DP와 강화학습의 차이

1. DP의 한계
   1. 계산 복잡도
   2. 차원의 저주
   3. 환경에 대한 완벽한 정보 필요

-> DP는 계산을 빠르게 하는 것일 뿐 “학습”을 하는 것이 아니다.

2. DP와 강화학습의 차이점

   ![image-20220222000308043](../../assets/images/2022-02-21-handwrite/image-20220222000308043.png)

## 예측

1. 몬테카를로 예측

   - 다이나믹 프로그래밍에서의 기존의 가치함수는 기댓값을 통해 계산
     $$
     v_π (s) = E_π [R_(t+1)+ γv_π (s_(t+1))|s_t= s]
     $$

   - 강화학습에서는 샘플링을 평균하여 가치함수 값을 추정<br/>

     (v_(n+1) 은 현재 받은 return과 이전에 받았던 return합을 더한 값의 평균)

     ![image-20220222000628662](../../assets/images/2022-02-21-handwrite/image-20220222000628662.png)

     ![image-20220222000706144](../../assets/images/2022-02-21-handwrite/image-20220222000706144.png)

   - 결국 몬테카를로 예측에서는 아래 식을 통해 가치함수를 업데이트 한다.

     ![image-20220222000852407](../../assets/images/2022-02-21-handwrite/image-20220222000852407.png)

   - a = 스텝사이즈(StepSize) : 업데이트 비율 ( learning rate )
   - **G (s) = 업데이트 목표** 
   - **a(G (s) - v(s)) = 업데이트의 크기**

   몬테카를로 예측의 단점 : 가치함수를 에피소드 마다 업데이트 한다.<br/>

   즉, 에피소드의 끝이 없거나, 길이가 긴 경우 부적합하다.

2. 시간차 예측

   - 가치함수를 매 타임스텝마다 업데이트 한다.

   - 가치함수 정의 : 
     $$
     𝑣_𝜋 (𝑠)= 𝐸_𝜋 [𝑅_(𝑡+1)+ 𝛾𝑣_  (𝑠_(𝑡+1) )│𝑠_𝑡= 𝑠]
     $$

   - 시간차 예측에서 가치함수 :

     ![image-20220222001303975](../../assets/images/2022-02-21-handwrite/image-20220222001303975.png)

   - 부트스트랩

     ![image-20220222001341394](../../assets/images/2022-02-21-handwrite/image-20220222001341394.png)

   - 몬테카를로 예측에서는 반환값 G(s)를 통해 업데이트 하지만, 시간차 예측에서는 현재 에이전트가 가지고 있는 v(s_(t+1) ) 값을 s_(t+1)의 가치함수로 가정하여 업데이트 한다
   - 업데이트 목표가 정확하지 않은 상황에서 가치함수를 업데이트 하는 것을 부트스트랩이라고 한다.

## SARSA

- SARSA = 시간차 제어 = 시간차 예측 + 탐욕 정책

- 정책 이터레이션(GPI)의 탐욕 정책 발전 :

  ![image-20220222001900618](../../assets/images/2022-02-21-handwrite/image-20220222001900618.png)

- 시간차 제어의 큐함수를 사용한 탐욕 정책 : 

  ![image-20220222001915782](../../assets/images/2022-02-21-handwrite/image-20220222001915782.png)

- 탐욕 정책에서 다음 상태의 가치함수를 보고 판단하는 것이 아니고, 현재 상태의 큐함수를 보고 판단하여 환경의 모델을 몰라도 행동을 선택할 수 있다.

- 시간차 예측에서 업데이트 하는 부분도 가치함수가 아니라 큐함수

  ![image-20220222001948111](../../assets/images/2022-02-21-handwrite/image-20220222001948111.png)<br>

  ![image-20220222002004348](../../assets/images/2022-02-21-handwrite/image-20220222002004348.png)

- 여기서 S,A,R, S,A 을 하나의 샘플로 사용하기 때문에 시간차 제어를 SARSA라고 한다.

- 하지만, 계속 탐욕 정책을 사용할 경우 지역최적점에 빠질 수 있으므로

  ![image-20220222002055363](../../assets/images/2022-02-21-handwrite/image-20220222002055363.png)

- ε의 확률로 무작위 행동을 선택한다.

- SARSA는 on-policy temporal-difference control 이기에  ε-탐욕 정책 사용시 잘못된 정책을 학습하여 갇힐 수 있다.

- 행동하는 대로 학습을 하기 때문에 다음 상태에서 다음 행동을 했을 때

   큐함수가 낮으면, 이후에 그 행동이 좋지 않다고 판단하여

   그 행동을 하지 않는 문제가 생긴다.


## Q-learning

- 때문에 Off-policy Temporal-Difference control을 사용 (대표적 : Q-learning)

- 행동하는 정책과 학습하는 정책을 따로 분리하며, 

  행동 정책으로 ε-탐욕 정책을 사용하고, 

  학습 정책으로는 다음 큐함수 중에서 최대 큐함수를 이용해서 현재 상태의 큐함수로 업데이트 함

  ![image-20220222010053814](../../assets/images/2022-02-21-handwrite/image-20220222010053814.png)

- 다음 상태에서 다음 행동을 해보는 것이 아니라 

- 다음 상태에서 가장 큰 큐함수를 가지고 업데이트

- 다음 상태에서 가장 큰 큐함수만 필요하기 때문에

-  샘플도 S,A,R,S 까지만 필요

  ![image-20220222010159438](../../assets/images/2022-02-21-handwrite/image-20220222010159438.png)

- SARSA와 Q-learning 비교

  <iframe width="560" height="315" src="https://www.youtube.com/embed/DTrHFyTraEA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

- SARSA는 몇번의 업데이트 이후 처음에서 3번째 상태로 가지 못하는 반면, Q-learning은 최적의 길을 찾아 감
- 결국 갇히지 않는다는 특성을 이용해 epsilon을 높여 높은 탐험성을 부여하면, 최적점을 찾기에도 유리함

## 근사 함수 

- 다이나믹 프로그램의 한계를 해결하고자 몬테카를로, 살사, 큐러닝 등을 고안

- 세 알고리즘은 model-free 이지만, 모든 상태에 대해 테이블을 만들어 놓고 테이블의 각 한에 큐 함수를 적는 ‘테이블 형태의 강화학습‘ 이다.
- 테이블 형태의 강화학습은 계산 복잡도와 차원의 저주 문제를 여전히 가짐
- 큐함수를 매개변수로 근사함으로 테이블을 사용하지 않아 해결 가능
- 기존의 산발적인 데이터를 데이터의 경향을 대표하는 함수로 표현하는 것을 의미
- 아래는 1차 함수 2차 함수 n함수로 근사한 예제이며, 딥러닝 발전 이후 대분의 근사함수로 인공 신경망을 사용![image-20220222011825883](../../assets/images/2022-02-21-handwrite/image-20220222011825883.png)
- 강화학습에서도 인공신경망을 사용

## 딥살사

- SARSA알고리즘을 이용하되 큐함수를 심층신경망으로 근사함
- 기존의 강화학습 코드처럼 업데이트 식을 작성할 필요가 없다. (오차함수 정의가 필요)

![image-20220222011936425](../../assets/images/2022-02-21-handwrite/image-20220222011936425.png)

<iframe width="560" height="315" src="https://www.youtube.com/embed/26_pG6qeneE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

![image-20220222013159252](../../assets/images/2022-02-21-handwrite/image-20220222013159252.png)

훈련 과정에서 Episode수에 따른 스코어

## 폴리시 그레디언트

- 지금까지의 강화학습 알고리즘은 가치 기반 강화학습으로 가치함수를 기반으로 행동을 선택하고, 업데이트하면서 학습한다. 
- 하지만, 정책 기반 강화학습은 가치함수를 토대로 행동을 결정하지 않고 상태에 따로 바로 행동을 선택한다. 
- 기존 정책은 상태마다 행동에 따라 확률 이 주어지지만 이제는 정책신경망이 정책을 대체한다.
- 정책신경망(정책을 근사하는 인공신경망)에서는 출력층의 활성함수가 Softmax함수 -> 정책의 정의가 바로 각 행동을 할 확률이기 때문에
- 정책을 최적화하는 목표함수(Objective Fuction)를 J라고 할 때

![image-20220222013311409](../../assets/images/2022-02-21-handwrite/image-20220222013311409.png)

