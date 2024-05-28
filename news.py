from bs4 import BeautifulSoup
import urllib3

#TODO: make button options on bot for which news site you want to visit
# credit: https://medium.com/@nishantsahoo/which-movie-should-i-watch-5c83a3c0f5b1
def get_title() -> str:
    url = "https://www.cnn.com/world"
    ourURL = urllib3.PoolManager().request('GET', url).data
    soup = BeautifulSoup(ourURL, "html.parser")
    #print(soup.title)
    lst = (soup.select('.layout--wide-left-balanced-2 .container__headline-text'))
    print(lst)
    result_str = "World News - CNN\n"
    for i in range(len(lst)):
        result_str += (f"{i}. {lst[i].text} \n")
    return result_str