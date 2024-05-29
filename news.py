from bs4 import BeautifulSoup
import urllib3

# credit: https://medium.com/@nishantsahoo/which-movie-should-i-watch-5c83a3c0f5b1
def get_cnn() -> str:
    url = "https://www.cnn.com/world"
    ourURL = urllib3.PoolManager().request('GET', url).data
    soup = BeautifulSoup(ourURL, "html.parser")
    lst = (soup.select('.layout--wide-left-balanced-2 .container__headline-text'))
    result_str = "10 World News Headlines - CNN\n"
    for i in range(min(len(lst), 10)):
        result_str += (f"{i}. {lst[i].text} \n")
    return result_str

def get_nbc() -> str:
    url = "https://www.nbcnews.com/world"
    ourURL = urllib3.PoolManager().request('GET', url).data
    soup = BeautifulSoup(ourURL, "html.parser")
    lst = soup.select('.relative .tease-card__headline')
    lst.extend(soup.select('.wide-tease-item__headline'))
    result_str = "10 World News Headlines - NBC\n"
    for i in range(min(len(lst), 10)):
        result_str += (f"{i}. {lst[i].text} \n")
    return result_str