# 조리 시간을 분 단위로 입력받아 시간과 분으로 변환하는 함수 만들기
def convert_minutes_to_hours(minutes):
    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}시간 {remaining_minutes}분"

# 예: 95분 -> 1시간 35분
user_input = input("조리 시간을 분 단위로 입력하세요: ")

try:
    minutes = int(user_input)
    result = convert_minutes_to_hours(minutes)
    print(result)
except ValueError:
    print("숫자만 입력해주세요.")