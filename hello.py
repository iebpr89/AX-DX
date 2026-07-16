# 숫자 리스트를 입력받아 평균을 계산하는 함수를 작성해줘     
def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)