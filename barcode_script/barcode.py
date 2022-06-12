import cv2;
import imutils
import numpy as np
from pyzbar.pyzbar import decode
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

class barcodeViz(object):
    def __init__ (self, image):
        self.image = image
    
    def getProductInfo(self, barcodeData):
        #getting barcode
        ff_options = Options()
        ff_options.add_argument("--headless")
        driver = webdriver.Firefox(executable_path = 'geckodriver', options= ff_options)

        URL = "https://www.barcodelookup.com/"+barcodeData
        #driver = webdriver.Chrome('chromedriver', options = chrome_options)
        driver.get(URL)
        product_name = driver.find_elements_by_xpath('//*[@id="product"]/section[2]/div[1]/div/div/div[2]/h4')
        product_name = product_name[0].text

        #getting amazon product
        product_string = product_name.replace(" ", "_")
        product_url = 'https://www.amazon.com/s?k='+product_string+ '&page=0'
        driver.get(product_url)
        product_picture_link = driver.find_element_by_xpath('//*[@id="search"]/div[1]/div[1]/div/span[3]/div[2]/div[2]/div/div/div/div/div/div[1]/div/div[2]/div/span/a/div/img')
        price_whole = driver.find_element_by_xpath('//*[@id="search"]/div[1]/div[1]/div/span[3]/div[2]/div[2]/div/div/div/div/div/div[2]/div/div/div[3]/div[1]/div/div[1]/div[2]/a/span[2]/span[2]/span[2]')
        price_frac = driver.find_element_by_xpath('//*[@id="search"]/div[1]/div[1]/div/span[3]/div[2]/div[2]/div/div/div/div/div/div[2]/div/div/div[3]/div[1]/div/div[1]/div[2]/a/span[2]/span[2]/span[3]')
        pic = product_picture_link.get_attribute('src')
        price = price_whole.text + "." + price_frac.text
        driver.quit()
        return ([pic, price])

    def getBarcode(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        barcode = decode(gray)
        barcodeData = ""
        if(len(barcode)<0):
            return ([])
        for obj in barcode:
            points = obj.polygon
            (x,y,w,h) = obj.rect
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(self.image, [pts], True, (0,255,0),3)
            barcodeData = obj.data.decode("utf-8")
            barcodeType = obj.type
            
        product_info = self.getProductInfo(barcodeData)
        return product_info

    def findBarcode(self):
        default = ({"image" :self.image, "foundBarcode": False})
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ddepth = cv2.CV_32F
        gradX = cv2.Sobel(gray, ddepth = ddepth, dx =1, dy= 0, ksize = -1)
        gradY = cv2.Sobel(gray, ddepth = ddepth, dx = 0, dy=1, ksize = -1)

        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        blurred = cv2.blur(gradient, (9,9))
        (_, thresh) = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        closed = cv2.erode(closed, None, iterations = 4)
        closed = cv2.dilate(closed, None, iterations = 4)

        cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if(len(cnts)<0):
           return default

        c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        rect = cv2.minAreaRect(c)
        if rect==0:
            return default
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        self.image= cv2.drawContours(self.image, [box], -1, (0,255, 0), 3)
        product_info = []
        #product_info = self.getBarcode()
        return ({"image" :self.image, "foundBarcode": True, "product_info": product_info})

vid = cv2.VideoCapture(0)
while(True):
    ret, frame = vid.read()
    br =barcodeViz(frame)
    info  = br.findBarcode()
    if info["foundBarcode"]:
        product_info = info["product_info"]
    frame = info["image"]
    #frame = cv2.putText(frame, product_info[0])
    #frame = cv2.putText(frame, product_info[1])
    cv2.imshow('frame', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

vid.release()
cv2.destroyAllWindows()