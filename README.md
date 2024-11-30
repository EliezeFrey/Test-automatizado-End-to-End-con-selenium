from selenium.webdriver.common.by import By
import unittest
from selenium import webdriver  # Asegúrate de tener instalado Selenium
from pyunitreport import HTMLTestRunner  # Asegúrate de tener instalado pyunitreport

base_url = "https://www.saucedemo.com/v1/"

class HomePage:
    def _init_(self, driver) -> None:
        self.driver = driver
        self.xpath_image_login = "/html/body/div[2]/div[1]"
        self.input_user = "user-name"
        self.input_password = "password"
        self.button = "login-button"
        self.xpath_page_init = ""
        self.xpath_error = ""

    def get_page_login(self):
        return self.driver.find_element(By.ID, self.button)


class TestHomePage(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.get(base_url)
        self.home_page = HomePage(self.driver)

    def test_login_button_present(self):
        """Verifica que el botón de login esté presente en la página."""
        login_button = self.home_page.get_page_login()
        self.assertIsNotNone(login_button)

    def tearDown(self):
        self.driver.quit()


if _name_ == '_main_':
    unittest.main(verbosity=2, testRunner=HTMLTestRunner(output='reports', report_name='Prueba'))
