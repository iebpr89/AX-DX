# """간단한 오븐 온도 추천 프로그램"""
#
# # 요리 종류와 추천 온도를 짝지어 저장합니다.
# temperature_guide = {
# 	"피자": 200,
# 	"쿠키": 180,
# 	"닭고기": 190,
# 	"생선": 170,
# }
#
# # 사용자에게 요리 종류를 입력받습니다.
# dish = input("요리 종류를 입력하세요 (피자/쿠키/닭고기/생선): ").strip()
#
# # 딕셔너리에서 입력한 요리의 온도를 찾습니다.
# recommended_temperature = temperature_guide.get(dish)
#
# # 온도 정보가 있으면 출력하고, 없으면 안내 문구를 출력합니다.
# if recommended_temperature is not None:
# 	print(f"추천 온도: {recommended_temperature}도")
# else:
# 	print("추천 온도 정보가 없습니다")

# 딕셔너리를 사용하는 간단한 버전
temperature_guide = {
	"피자": "추천 온도: 200도",
	"쿠키": "추천 온도: 180도",
	"닭고기": "추천 온도: 190도",
	"생선": "추천 온도: 170도",
}

dish = input("요리 종류를 입력하세요 (피자/쿠키/닭고기/생선): ").strip()
print(temperature_guide.get(dish, "추천 온도 정보가 없습니다"))
