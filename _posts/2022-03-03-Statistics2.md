---
title: 탐색적 데이터 분석
date: 2022-03-03
category: study
tags:
    - statistics
toc: true
author_profile: false
sidebar:
  nav: "docs"


---

## 1. 정형화된 데이터의 요소

- numeric : 숫자를 이용해 표현할 수 있는 데이터
- continuous : 일정 범위 안에서 어떤 값이든 취할 수 있는 데이터
- discrete : 횟수와 같은 정수 값만 취할 수 있는 데이터
- dategorical : 가능한 범주 안의 값만을 취하는 데이터
- binary : 두 개의 값 만을 갖는 범주형 데이터
- ordinal : 순위가 있는 범주형 데이터

## 2. 테이블 데이터

- data frame : 숫자를 이용해 표현할 수 있는 데이터
- feature : 일정 범위 안에서 어떤 값이든 취할 수 있는 데이터
- outcome : 횟수와 같은 정수 값만 취할 수 있는 데이터
- record : 가능한 범주 안의 값만을 취하는 데이터

##  3. 위치 추정

* mean : 모든 값의 총합을 개수로 나눈 값

  

* weighted mean : 가중치를 곱한 값의 총합을 가중치의 총합으로 나눈 값

  

* median : 중간값

* percentile : 백분위

* weighted median : 가중 치 값을 위에서부터 더할 때, 총합의 중간이 위치하는 데이터 값

* trimmed mean : 정해진 개수의 극단값을 제외한 나머지 값들을 평균

* robust : 극단값들에 민감하지 않다는 것을 의미

* outlier : 대부분의 값과 다른 데이터 값

## 4. 변이 추정

* deviation : 관측값과 위치 추정값 사이의 차이
* variance : 평균과의 편차를 제곱한 값들의 합을 (데이터 수-1)로 나눈 값
* standard deviation : 분산의 제곱근
* mean absolute deviation : 평균과의 편차의 절댓값의 평균
* MAD : 중간값과 편차의 절댓값의 중간값
* range : 최대, 최소의 차이
* order statistics : 최소에서 최대까지 정렬된 데이터 값에 따른 계량형
* percentile : 어떤 값들의 퍼센트가 이 값 혹은 더 작은 값을 갖고, 퍼센트가 이값 혹은 더 큰 값을 갖도록 하는 값
* interquartile range : 75번째 백분위수와 25번째 백분위수 사이의 차이

