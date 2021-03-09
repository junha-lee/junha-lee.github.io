---
title: "테마를 이용해 github.io 블로그 구축하기"
categories: 
  - blog
last_modified_at: 2020-01-14T15:00:00+09:00
toc: true
---

Intro
--------------------------
이 페이지는 본래 블로그 구축을 처음 했을 때 테스트 페이지로 포스팅했던 것인데, 그 과정에서 했던 깃 사용에 익숙하지 않았던 제가 했던ㅠㅠ 수많은 삽질 이후.. 정리해서 누군가에게는 도움이 될만한 자료로 남겨두려고 수정합니다.<br/>

깃은 개발자들의 성지라고 할 수 있죠!<br/>
요즘 대외활동 동아리, 취업 자소서나 포폴 작성 제출할 경우에도 깃 주소를 기재하도록 양식을 주기도 할 뿐더러 그에 따라 컨퍼런스나 주니어 개발자 격언 주제의 강연을 들을 때에도 항상 듣는 말이 **깃 관리! 블로그 관리!** 입니다.<br/>
본래 깃은 버전관리 목적의 툴이다보니 매일매일 개발과정에 대해 커밋하는게 쉽지 않기도 하고, 이론 공부한 날까지도 공부했다는 기록을 남길 수 있으니 깃 페이지 블로그 관리로 열정과 ㅎㅎ 성실함을 증명해낼 수 있다면 참 좋을 거라고 생각했습니다.
그리고 동계 방학이 되다보니 깃허브 잔디찍기 프로젝트가 개발자들 사이에 유행을 하기도 해서 제가 연구실장으로 임하고 있는 소속 대학교 연구실 친구들에게 개강까지 깃 1일 1커밋 D-Day 50일 프로젝트를 과제로 줬답니다.<br/>

하지만 진입장벽이 다소 있는 편이다보니 연구실부원 중에는 1학년 신입 학부생들도 있는데 제가 했던 삽질을 그대로 온전히 겪게 하는 것보다는 어느 정도의 공부 환경은 먼저 만들어주는 것이 좋을 것 같아 레퍼런스로 참고하라고 이 글을 작성하기도 합니다.<br/>

개인적으로는 제가 구글링을 못하는 것인지 모르겠지만 여러 블로그를 거쳐가며 삽질의 완성으로 이 홈페이지를 운영하기까지를 포스팅하면 좋을 것 같았어요! 제가 그렇게 찾아 헤맸지만 잘 나오지 않아던! 그런 글! 우여곡절 끝에 알아낸 꼼수를 정리해두면 꼭 저의 연구실 동료들 뿐아니라 얼마전의 저와 같은 상황 속에 계시는 누군가에게는 도움이 되지 않을까... 생각했습니다ㅎㅎ<br/>

서론이 길었습니다. 말 그대로 '꼼수'라서 제 방법이 아주 좋은 방법이라고 자부하긴 어렵다는 점 다시 한 번 말씀드리고 싶습니다. ㅎ ㅎ<br/><br/>

제 방식은 간단히 설명하자면<br/>
1. 블로그 테마를 고르고(css 없이 예쁜 디자인을 쓸 수 있다는 장점)<br/>
2. 로컬에 블로그 프로젝트 저장소를 생성한 뒤<br/>
3. 원격 레포지토리를 생성하여 연동시켜주기<br/>
로 진행됩니다.<br/>



Select Theme & download
--------------------------
저는 minial mistakes 테마를 사용하였습니다.<br/>
로컬 PC에 zip파일을 다운로드 받습니다.<br/>
(출처 : [minial mistakes](https://github.com/mmistakes/minimal-mistakes))
{% raw %} <img src="https://ohjinjin.github.io/assets/images/20200114blog/capture1.JPG" alt=""> {% endraw %}
{% raw %} <img src="https://ohjinjin.github.io/assets/images/20200114blog/capture2.JPG" alt=""> {% endraw %}

사실 블로그를 뚝딱 구현하기에는 folk해오는 것이 더욱 편리하지만, 여러분 깃 계정 overview에 예쁘게 잔디를 깔기 위해서는 경험상 이 방법이 바람직합니다!


Create Remote Repository
--------------------------
깃허브 원격 레포지토리를 새로 생성합니다.<br/>
{% raw %} <img src="https://ohjinjin.github.io/assets/images/20200114blog/capture3.JPG" alt=""> {% endraw %}

깃페이지는 계정당 하나만을 제공합니다.<br/>
저의 경우에는 이미 깃페이지를 사용하고 있기 때문에 중복되었다는 경고(The repository username.github.io already exists on this account)를 띄워주지만 새롭게 만드는 분이시라면 정상적으로 create가 가능하실 겁니다.<br/>

깃페이지 저장소로 사용할 원격 레포지토리의 이름은 반드시 username.github.io의 형식으로 작성해주셔야만 합니다.<br/>
그리고 이 원격 레포지토리명은 모든 작업이 끝났을때 여러분들이 사용하실 블로그의 url이 됩니다!<br/>

Description은 마음에 드시는대로 달아주셔도 됩니다. 저는 ohjinjin's DevLog라고 달았답니다.<br/>
readme 마크다운파일은 생성하지 않겠습니다. 이따 zip 해제해보면 아시겠지만 이미 있거든요 ㅎㅎ<br/>


Create Local Repository
--------------------------
용량이 넉넉한 로컬 저장소에 같은 이름 'username.github.io'의 디렉토리를 생성합니다. 저는 D:\에 만들었어요!<br/>

아까 다운로드 받았던 테마 압축파일을 해당 로컬 디렉토리가 **최상단 디렉토리**가 되도록 **압축해제**합니다.<br/>
{% raw %} <img src="https://ohjinjin.github.io/assets/images/20200114blog/capture4.JPG" alt=""> {% endraw %}


Connect Local-Remote Repo
--------------------------
최상위 디렉토리를 열어 둔 상태에서 파일 탐색기 내 빈 공간에 마우스 우클릭하여 **git bash here**을 클릭합니다.<br/>
{% raw %} <img src="https://ohjinjin.github.io/assets/images/20200114blog/capture5.JPG" alt=""> {% endraw %}

만약 저게 보이지 않으신다면.. git을 처음 사용하시는 분이시리라 추측합니다. [git download](https://coding-factory.tistory.com/245) 받으셔야합니다!<br/>
깃 다운로드와 관련하여 제가 참고했던 잘 정리된 포스트의 링크 걸어드리겠습니다. 16번까지 꼭 진행해주시기 바랍니다.<br/>

git bash shell이 열렸으면 차례로 아래 명령어를 입력하시면됩니다.<br/>
괄호 안에는 해당 명령어에 대한 간단한 설명이 있습니다.<br/>
>git init<br/>
(초기화)<br/>

>git status<br/>
(깃은 버전관리툴로, 현재까지의 변동사항을 보여줍니다.)<br/>

>git add .<br/>
(stage area로 현재 지점의 수정사항 모두를 올려줍니다. 혹시 몇몇의 경고가 뜰 경우 무시하셔도 문제없이 진행됩니다!)<br/>

>git status<br/>
(다시 한 번 변동사항을 조회해봅니다.)<br/>

>git commit -m "first commit"<br/>
(커밋 메시지(first commit)와 함께 커밋합니다.)<br/><br/><br/>

>git status<br/>
(다시 한 번 변동사항을 조회해봅니다. 커밋을 한 직후이므로 이제 수정내역이 없다고 나올 거에요.)<br/>

>git remote add origin https://github.com/username/username.github.io.git<br/>
(로컬 레포지토리와 원격 레포지토리를 연동시킵니다. username 부분은 여러분의 깃 계정으로 바꿔 입력하시면 됩니다.)<br/>

>git push -u origin master<br/>
(모든 commit 이력에 대해 원격 레포지토리로 push 업데이트 해줍니다.)<br/>

(처음 깃을 설치하신 경우에는 최초 한 번 로그인창이 뜰 겁니다. 그럴 경우 그냥 로그인만 해주시면 됩니다.)

오류없이 진행되었다면 원격 레포지토리를 새로고침해서 확인해보세요!
{% raw %} <img src="https://ohjinjin.github.io/assets/images/20200114blog/capture6.JPG" alt=""> {% endraw %}

이렇게 되었다면 성공적입니다.<br/>
처음 깃 페이지가 생성될 경우에는 조금의 시간이 걸립니다.<br/>
브라우저 주소창에 https://username.github.io로 접속해보시면 곧 초기 화면이 보이실 거에요.<br/><br/>


Post
--------------------------
저장소를 자세히보면 여러분들 것과 제 것이 조금 다른 부분이 있죠? 예를 들자면 여러분들 로컬 및 원격 저장소에는 최상위 폴더 내 _posts라는 디렉토리가 안보이실 겁니다.<br/>
참고로 _posts 폴더에는 각 게시물에 대한 마크다운 파일들을 저장할 겁니다.<br/>

원본 테마를 그대로 다운받으셨기때문에 여러분들께서 딱히 사용하지 않을 파일들까지 섞여있답니다.<br/>
그러므로 첫 포스트를 게재하기 전 우선 불필요한 파일들을 로컬 디렉토리에서 삭제해줍시다.<br/>
* .editorconfig
* .gitattributes
* .github
* /docs
* /test
* CHANGELOG.md
* minimal-mistakes-jekyll.gemspec
* README.md
* screenshot-layouts.png
* screenshot.png

이 목록을 전부 삭제하셨다면 첫 포스트를 게시해봅시다.<br/>

기억하셔야할게 지금부터 모든 수정 워크플로우는 '로컬에서 작업\>원격레포에 커밋/푸시하여 반영'하는 순서로 갑니다.<br/>

로컬 최상위인 username.github.io 디렉토리 안에 **_posts**라는 이름의 폴더를 생성해줍니다.<br/>
그리고 새 메모장파일을 열어 _posts 디렉토리 안에 다른 이름으로 저장합니다.<br/>
이 때 파일 형식을 **모든 파일**로, 파일 제목은 YYYY\-MM\-DD\-제목.md으로 형식을 준수하여 생성해주시면됩니다.<br/>

{% raw %} <img src="https://ohjinjin.github.io/assets/images/20200114blog/capture7.JPG" alt=""> {% endraw %}

해당 마크다운의 파일 내용으로는 가장 먼저<br/>
아래 포맷의 마크다운 헤더를 작성하는 것으로 시작됩니다.<br/>
>\-\-\-<br/>
>title: "깃페이지를 이용하여 블로그 구축하기"<br/>
>categories: <br/>
>  \- blogging<br/>
>last_modified_at: 2020-01-14T14:00:00\+09:00<br/>
>toc: true<br/>
>\-\-\-

<br/>
(https://mmistakes.github.io/minimal-mistakes/에서 여러 샘플을 참고하시면 여러 헤더 예제가 확인됩니다.)

헤더를 작성하셨다면 마지막 줄인 '\-\-\-' 밑으로 개행하셔서 게시물 내용을 작성해주세요.<br/>
간단한 테스트를 위해서 'this is a test page' 정도가 좋겠습니다!<br/>

참고로 추후 잘 정리된 예쁜 게시물을 만들고자 하신다면 [minimal-mistakes테마의 샘플들](https://mmistakes.github.io/minimal-mistakes/) 및 [마크다운 문법 예제](http://guswnsxodlf.github.io/how-to-write-markdown) 등을 참고해주시면 됩니다.<br/>

이렇게 로컬에서 프로젝트 수정을 했는데요, 과연 원격 저장소와 실제 여러분의 깃 페이지에까지 반영이 되었을까요?<br/>
그럴리가 없겠지요!<br/>

로컬에서 작업한 변동사항을 commit해주고, push까지 해주셔야만 반영된다는 점 기억해주세요!<br/>

다시 한 번 git bash창을 열어보겠습니다.<br/>
로컬 레포지토리 최상위 폴더를 파일탐색기로 열어 빈 공간에 마우스 우클릭하셔서 git bash here를 클릭합니다.<br/>

아까 초기 설정을 이미 했기 때문에 이번엔 명령어를 살짝 다르게 입력할 겁니다.
>git add .<br/>
>git commit -m "created test post"<br/>
>git push -u origin master<br/>

이거면 됩니다!

원격 레포지토리와 여러분들의 깃 페이지를 새로고침해서 확인해보시면 됩니다!<br/>
<br/>

이렇게 하루 하루 공부하신 것에 대해 정리한 연구노트나 개발일지를 작성하여 잔디찍기 프로젝트에 도전해보세요! ㅎㅎ<br/>

(아래에는 블로그 첫 페이지를 커스텀하는 내용이 나옵니다.)<br/>

Custom Main Page
--------------------------
마찬가지로 이 과정도 로컬 레포지토리에서 작업하고 원격 레포지토리로 커밋/푸시해주세요.<br/>

레포 최상위 디렉토리에 있는 _config.yml파일을 열어 title이나, site author등을 원하시는 대로 수정하시면 됩니다.<br/>
또 sns도 링크 걸 수 있습니다.<br/>
참고로 야믈은 객체 표현 방식 중 하나라고 보시면되고, 또 다른 예로는 json, xml 등이 있습니다. 그렇기 때문에 이 파일을 수정할 때에도 역시 메모장으로 연결해서 열어보시면 됩니다.<br/>

제 블로그처럼 우측 상단에 Quick Start 페이지로 이어지는 링크가 아닌 About페이지나 Category 페이지로의 링크를 만들고 싶으시다면 최상위 폴더에서 _data라는 폴더 내에 있는 navigation.yml을 수정해주시면 됩니다!<br/>
제 블로그와 관련하여 자세한 코드가 궁금하신 분들께서는 [이곳](https://github.com/ohjinjin/ohjinjin.github.io/blob/master/_data/navigation.yml)을 클릭해주세요.<br/>

또한 404상태임을 알리는 페이지부터, 방금 네비게이션 바에 설정한 about과 category도 컨텐츠들을 띄우고 싶으시다면 최상위 폴더에서 _pages 디렉토리로 들어가 404.md, about.md 그리고 category\-archive.md 파일들을 수정해주시면됩니다.<br/>
제 블로그와 관련하여 자세한 코드가 궁금하신 분들께서는 [이곳](https://github.com/ohjinjin/ohjinjin.github.io/tree/master/_pages)을 클릭해주세요.<br/>

기본적인 내용만을 글에 담았으나 혹시 메모장이 불편하시다면 마크다운 파일을 작성하실 때에도 그렇고 에디터 편한 거 다운받아서 사용하시면 됩니다!<br/>


제가 어떻게 변경했는지 혹시나 궁금하시다면 [여기서](https://github.com/ohjinjin/ohjinjin.github.io/blob/master/_config.yml) 야믈 파일 내용을 참고하시면 됩니다.<br/><br/>

모든 과정을 다 마치셨다면 깃허브 본인계정 overview로 돌아가 여러분께서 스스로 심은 잔디를 봐보세요! 뿌듯함이 밀려오지 않습니까 ㅎ   ㅎ
감사합니다!


개인이 공부하고 포스팅하는 블로그입니다. 작성한 글 중 오류나 틀린 부분이 있을 경우 과감한 지적 환영합니다!<br/><br/>