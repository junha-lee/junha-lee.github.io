---
title: coal crawling
date: 2020-07-18
category: coal
tags:
    - web crawling
    - Selenium
    - coal
    - IoT
toc: true
author_profile: false
sidebar:
  nav: "docs"
---





### 목표 데이터 정의

---

![coal_crawl](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/data.png)

### Selenium을 이용한 스크래핑
---


```python
#크롬의 해당 이미지 url 접속 코드

Search_term = 'Bituminous coal'
url = https://www.google.co.in/search?q=+search_term+"&tbm=isch“
browser = webdriver.Chrome('chromedriver.exe')
browser.get(url)

#스크롤 코드
browser.execute_script('window.scrollBy(0,10000)')

#해당 클래스 이름을 가진 요소 찾는 코드
browser.find_elements_by_class_name("rg_i")

#이미지로 저장하는 코드
el.screenshot(str(idx) + ".png")

```

### 스크래핑 결과
---

![coal_crawl](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/bituminous_coal.png)

### Dataset
---

연갈탄과 유연탄 사진

Trainset – 연갈탄 20개, 유연탄 120개

Testset –  연갈탄 6개, 유연탄 14개

validationset –  연갈탄 3개, 유연탄 7개


### 마치며
---

석탄 분류기 제조를 위해 연갈탄 및 유연탄 데이터를 수집 해 봤습니다.

읽어주셔서 감사합니다.
