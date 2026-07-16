// localStorage에 저장할 때 사용할 고정 키 이름입니다.
const TODO_STORAGE_KEY = "todoAppItems";

// 현재 할 일 목록 데이터를 메모리에 보관합니다.
let todoItems = [];

// 자주 사용하는 화면 요소를 변수로 가져옵니다.
const todoInputElement = document.getElementById("todoInput");
const addTodoButtonElement = document.getElementById("addTodoButton");
const clearAllButtonElement = document.getElementById("clearAllButton");
const todoListElement = document.getElementById("todoList");
const remainingCountTextElement = document.getElementById("remainingCountText");

// localStorage에서 저장된 할 일 목록을 읽어옵니다.
function loadTodoItemsFromLocalStorage() {
    const savedDataText = localStorage.getItem(TODO_STORAGE_KEY);

    // 저장된 값이 없으면 빈 배열로 시작합니다.
    if (!savedDataText) {
        todoItems = [];
        return;
    }

    try {
        const parsedData = JSON.parse(savedDataText);
        todoItems = Array.isArray(parsedData) ? parsedData : [];
    } catch (error) {
        // JSON 파싱에 실패하면 손상된 데이터라고 보고 초기화합니다.
        todoItems = [];
    }
}

// 현재 todoItems 배열을 localStorage에 저장합니다.
function saveTodoItemsToLocalStorage() {
    localStorage.setItem(TODO_STORAGE_KEY, JSON.stringify(todoItems));
}

// 새 할 일 객체를 생성해서 반환합니다.
function createTodoItem(text) {
    return {
        id: Date.now().toString(),
        text,
        completed: false,
    };
}

// 할 일 목록 화면을 현재 데이터 상태로 다시 그립니다.
function renderTodoList() {
    todoListElement.innerHTML = "";
    updateRemainingTodoCountText();

    // 할 일이 하나도 없을 때 안내 문구를 보여줍니다.
    if (todoItems.length === 0) {
        const emptyMessageElement = document.createElement("li");
        emptyMessageElement.className = "empty-message";
        emptyMessageElement.textContent = "아직 등록된 할 일이 없습니다.";
        todoListElement.appendChild(emptyMessageElement);
        return;
    }

    // 할 일 객체 하나당 화면 요소를 만들어 목록에 추가합니다.
    todoItems.forEach((todoItem) => {
        const listItemElement = createTodoListItemElement(todoItem);
        todoListElement.appendChild(listItemElement);
    });
}

// 완료되지 않은 할 일 개수를 계산해 화면에 보여줍니다.
function updateRemainingTodoCountText() {
    const remainingCount = todoItems.filter((todoItem) => !todoItem.completed).length;
    remainingCountTextElement.textContent = `남은 할 일: ${remainingCount}개`;
}

// 할 일 한 줄에 해당하는 li 요소를 만들어 반환합니다.
function createTodoListItemElement(todoItem) {
    const itemElement = document.createElement("li");
    itemElement.className = "todo-item";

    if (todoItem.completed) {
        itemElement.classList.add("completed");
    }

    const checkboxElement = document.createElement("input");
    checkboxElement.type = "checkbox";
    checkboxElement.checked = todoItem.completed;
    checkboxElement.addEventListener("change", () => {
        toggleTodoItemCompleted(todoItem.id);
    });

    const textElement = document.createElement("span");
    textElement.className = "todo-text";
    textElement.textContent = todoItem.text;

    const deleteButtonElement = document.createElement("button");
    deleteButtonElement.className = "todo-delete-button";
    deleteButtonElement.textContent = "삭제";
    deleteButtonElement.addEventListener("click", () => {
        deleteTodoItem(todoItem.id);
    });

    itemElement.appendChild(checkboxElement);
    itemElement.appendChild(textElement);
    itemElement.appendChild(deleteButtonElement);

    return itemElement;
}

// 입력창 텍스트를 읽어 새 할 일을 추가합니다.
function addTodoItemFromInput() {
    const inputText = todoInputElement.value.trim();

    // 빈 문자열은 저장하지 않도록 막아줍니다.
    if (inputText === "") {
        alert("할 일을 입력해주세요.");
        todoInputElement.focus();
        return;
    }

    const newTodoItem = createTodoItem(inputText);
    todoItems.push(newTodoItem);

    saveTodoItemsToLocalStorage();
    renderTodoList();

    todoInputElement.value = "";
    todoInputElement.focus();
}

// 특정 id를 가진 할 일의 완료 여부를 반전시킵니다.
function toggleTodoItemCompleted(todoItemId) {
    todoItems = todoItems.map((todoItem) => {
        if (todoItem.id === todoItemId) {
            return {
                ...todoItem,
                completed: !todoItem.completed,
            };
        }

        return todoItem;
    });

    saveTodoItemsToLocalStorage();
    renderTodoList();
}

// 특정 id를 가진 할 일을 목록에서 제거합니다.
function deleteTodoItem(todoItemId) {
    todoItems = todoItems.filter((todoItem) => todoItem.id !== todoItemId);

    saveTodoItemsToLocalStorage();
    renderTodoList();
}

// 전체 할 일을 모두 삭제합니다.
function clearAllTodoItems() {
    if (todoItems.length === 0) {
        alert("삭제할 항목이 없습니다.");
        return;
    }

    const isConfirmed = confirm("정말 전체 삭제하시겠습니까?");
    if (!isConfirmed) {
        return;
    }

    todoItems = [];
    saveTodoItemsToLocalStorage();
    renderTodoList();
}

// 앱 시작 시 초기 데이터와 이벤트를 연결합니다.
function initializeTodoApp() {
    loadTodoItemsFromLocalStorage();
    renderTodoList();

    addTodoButtonElement.addEventListener("click", addTodoItemFromInput);

    // 입력창에서 Enter 키를 누르면 추가 버튼과 같은 동작을 수행합니다.
    todoInputElement.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            addTodoItemFromInput();
        }
    });

    clearAllButtonElement.addEventListener("click", clearAllTodoItems);
}

initializeTodoApp();
