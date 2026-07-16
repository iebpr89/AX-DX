# 빌트인 오븐 설치 점검 체크리스트 프로그램

# 점검해야 할 항목들을 리스트로 저장합니다.
checklist_items = [
	"가구장 규격 확인",
	"전원 사양 확인",
	"환기 공간 확보",
	"수평 설치 확인",
	"도어 개폐 간섭 확인",
	"설치 후 시운전 완료",
]

# 모든 항목이 통과되었다고 가정하고 시작합니다.
all_passed = True

# n으로 답한 항목을 따로 모아둘 리스트입니다.
failed_items = []

# 리스트에 있는 각 항목을 하나씩 사용자에게 물어봅니다.
for item in checklist_items:
	while True:
		# y 또는 n 입력을 받습니다. (대문자 입력도 소문자로 바꿉니다.)
		answer = input(f"{item} (y/n): ").strip().lower()

		# 올바른 입력인지 먼저 확인합니다.
		if answer in ("y", "n"):
			break

		print("y 또는 n만 입력해주세요.")

	# 하나라도 n이면 최종 결과는 재점검 필요입니다.
	if answer == "n":
		all_passed = False
		failed_items.append(item)

# 모든 항목 결과를 바탕으로 최종 메시지를 출력합니다.
if all_passed:
	print("설치 완료 가능")
else:
	print("재점검 필요")
	print("n으로 입력한 항목:")
	for failed_item in failed_items:
		print(f"- {failed_item}")
