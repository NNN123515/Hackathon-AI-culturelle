const homeScreen = document.getElementById("homeScreen");
const answerScreen = document.getElementById("answerScreen");

const chatMessages = document.getElementById("chatMessages");

function addQuestion(question) {

    const turn = document.createElement("div");
    turn.className = "chat-turn";

    // Question utilisateur
    const userMessage = document.createElement("div");
    userMessage.className = "user-message";

    const messageText = document.createElement("p");
    messageText.textContent = question;

    userMessage.appendChild(messageText);

    // Réponses des deux IA
    const responses = document.createElement("div");
    responses.className = "ai-responses";

    responses.innerHTML = `
        <div class="ai-response">
            <h3>Réponse 1</h3>
            <div class="ai-response-content"
                 id="response-one-${Date.now()}">
            </div>
        </div>

        <div class="ai-response">
            <h3>Réponse 2</h3>
            <div class="ai-response-content"
                 id="response-two-${Date.now()}">
            </div>
        </div>
    `;

    turn.appendChild(userMessage);
    turn.appendChild(responses);

    chatMessages.appendChild(turn);

    // Descendre automatiquement vers le dernier message
    const chatContainer = document.querySelector(".chat-container");

    chatContainer.scrollTo({
        top: chatContainer.scrollHeight,
        behavior: "smooth"
    });

    // Les deux réponses seront ajoutées ici via les API.
}


// Première question
document.getElementById("questionForm")
.addEventListener("submit", (e) => {

    e.preventDefault();

    const input = document.getElementById("question");
    const question = input.value.trim();

    if (!question) return;

    homeScreen.hidden = true;
    answerScreen.hidden = false;

    addQuestion(question);

    input.value = "";
});


// Questions suivantes
document.getElementById("followupForm")
.addEventListener("submit", (e) => {

    e.preventDefault();

    const input = document.getElementById("followupInput");
    const question = input.value.trim();

    if (!question) return;

    addQuestion(question);

    input.value = "";
});
