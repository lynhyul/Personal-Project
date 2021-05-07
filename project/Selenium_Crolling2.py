from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

# selenium에서 사용할 웹 드라이버 절대 경로 정보
chromedriver = 'project/chromedriver.exe'
# selenum의 webdriver에 앞서 설치한 chromedirver를 연동한다.
driver = webdriver.Chrome(chromedriver)
# driver로 특정 페이지를 크롤링한다.
driver.get('https://www.shutterstock.com/ko/search?pl=PPC_NAV_KR_IG&kw=%EA%B0%95%EC%95%84%EC%A7%80%EC%9D%B4%EB%AF%B8%EC%A7%80') # google화면 띄우기
elem = driver.find_element_by_name("searchterm") # 검색창에 대한 elem
elem.send_keys("1024x1024 이미지")  # "q"라는 elem(검색창에)에 '비글'이라는 단어를 입력하게 하겠다.
elem.send_keys(Keys.RETURN)
time.sleep(2)


SCROLL_PAUSE_TIME = 3

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

for i in range(2):
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)
    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        try :
            element = driver.find_element_by_xpath("/html/body/div[2]/div[2]/div/div[2]/div[2]/main/div/div[3]/div[2]/div/div/a[2]")
            driver.execute_script("arguments[0].click();", element)
        except :
            break
    last_height = new_height 



images = driver.find_elements_by_css_selector(".z_h_9d80b.z_h_2f2f0") # 이미지에 대한 class name이 있는데

print(images)                                                    # class name을 통해서 이미지 지정(커서)
count = 1 
for image in images :
    try :
        image.click()   # 이미지를 클릭하겠다.
        time.sleep(1)
        imgUrl = driver.find_element_by_css_selector(".m_l_c4504").get_attribute("src")    
        urllib.request.urlretrieve(imgUrl, "../../data/image/super/"+ str(count)+ ".jpg")
        count = count + 1
        driver.back('https://www.shutterstock.com/ko/search/corgi?')
    except :
        pass
driver.close()
print("+" * 100)
print(driver.title)   # 크롤링한 페이지의 title 정보
print(driver.current_url)  # 현재 크롤링된 페이지의 url
print("바이크 브랜드 크롤링")
print("-" * 100)