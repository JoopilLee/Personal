from datetime import datetime, timedelta
import math
class ParkingFeeCalculator:
    def __init__(self, car_number, entry_time, exit_time):
        self.car_number = car_number
        self.entry_time = datetime.strptime(entry_time, "%Y-%m-%d %H:%M")
        self.exit_time = datetime.strptime(exit_time, "%Y-%m-%d %H:%M")

    def calculate_initial_fee(self):
        weekday = self.entry_time.weekday()
        entry_hour = self.entry_time.hour
        entry_minute = self.entry_time.minute

        if (weekday < 5 and (((entry_hour == 7 and entry_minute >= 30) or (entry_hour == 9 and entry_minute <= 30)) 
                             or (entry_hour >= 18 and entry_hour < 20))):
            initial_fee = 3000  # 평일 츨・퇴근 시간 입차 후 ~30분 3000원
        else:
            initial_fee = 2000  # 일반 입차 후 ~30분 2000원
        return initial_fee

    def calculate_additional_fee(self):
        duration = self.exit_time - self.entry_time
        duration_minutes = duration.total_seconds() // 60 

        if duration_minutes <= 30:
            additional_fee = 0
        elif duration_minutes <= 120:
            duration_minutes -= 30
            additional_fee = 500 * (math.floor(duration_minutes // 10))  # 2시간까지 10분당 500원씩 할증
        else:
            duration_minutes -= 120
            additional_fee = (500 * 9) + (1000 * (math.floor(duration_minutes // 10)))  # 2시간 이후 10분당 1000원씩 할증
        return additional_fee

    def calculate_parking_fee(self):
        if self.exit_time <= self.entry_time + timedelta(minutes=10):
            return 0 
        initial_fee = self.calculate_initial_fee()
        additional_fee = self.calculate_additional_fee()
        return int(initial_fee + additional_fee)

    def __str__(self):
        fee = self.calculate_parking_fee()
        return f" 차량 번호: {self.car_number}\n 입차시간 : {self.entry_time} \n 출차시간 : {self.exit_time} \n 주차요금: {fee:,}원 \n 좋은 하루 되세요!"

class AlumniFeeCalculator(ParkingFeeCalculator):
    def __init__(self, car_number, entry_time, exit_time, alumni_name):
        super().__init__(car_number, entry_time, exit_time)
        self.alumni_name = alumni_name

    def Alumni_parking_fee(self):
        duration = self.exit_time - self.entry_time
        duration_minutes = duration.total_seconds() // 60
        
        if self.entry_time.weekday() >= 5:
            if duration_minutes <= 300:
                 alumni_fee = 0  # 졸업생은 5시간까지 주말 무료주차
            else:
                alumni_fee = int(((duration_minutes - 300) // 10) * 1000)
        else:
            alumni_fee = super().calculate_parking_fee()
        return alumni_fee
    def __str__(self):
        alumni = self.Alumni_parking_fee()
        return f" {self.alumni_name} 동문님 안녕하세요! \n 차량 번호 : {self.car_number} \n 입차시간 : {self.entry_time} \n 출차시간 : {self.exit_time} \n 주차요금: {alumni:,}원 입니다.\n 좋은 하루 되세요!"

# 테스트 코드 # 
car_number_input = input('차량 번호를 입력하세요: ')
entry_time_input = input("입차 시간을 입력하세요 (YYYY-MM-DD HH:MM): ")
exit_time_input = input("출차 시간을 입력하세요 (YYYY-MM-DD HH:MM): ")
alumni_name = input("성함을 입력하세요: ")
alumni_database = ['봉준호','나영석','박지성','손흥민','류현진','손연재','최민정']

if alumni_name in alumni_database:
    calculator = AlumniFeeCalculator(car_number_input, entry_time_input, exit_time_input, alumni_name)
else:
    calculator = ParkingFeeCalculator(car_number_input, entry_time_input, exit_time_input)
print(calculator)