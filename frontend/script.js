const homeScreen = document.getElementById("homeScreen");
const answerScreen = document.getElementById("answerScreen");
const chatMessages = document.getElementById("chatMessages");
const API_URL = window.location.origin;





async function checkAPI() {
    const response = await fetch(`${API_URL}/health`);
    if (!response.ok) {
        throw new Error("L'API Elluna ne répond pas.");
    }

    return await response.json();
}


async function getCurrentModel() {
    const response = await fetch(`${API_URL}/get_model`);
    if (!response.ok) throw new Error("Impossible de récupérer le modèle.");

    return await response.json();
}

// Récupérer les modèles disponibles
async function getLocalModels() {
    const response = await fetch(`${API_URL}/models/local`);
    if (!response.ok) throw new Error("Impossible de récupérer les modèles locaux.");

    return await response.json();
}

async function getApiModels() {
    const response = await fetch(`${API_URL}/models/api`);
    if (!response.ok) throw new Error("Impossible de récupérer les modèles API.");

    return await response.json();
}

// Changer le modèle actif
async function changeModel(modelName) {
    const params = new URLSearchParams({
        new_model: modelName
    });

    const response = await fetch(
        `${API_URL}/change_model?${params}`,
        { method: "POST" }
    );

    if (!response.ok) throw new Error("Le changement de modèle a échoué.");

    return await response.json();
}

// Initialiser le sélecteur de modèles
async function initModelSelector() {
    const select = document.getElementById("modelSelect");
    const status = document.getElementById("modelStatus");

    try {
        // Récupérer les modèles disponibles et le modèle actif
        const [localModels, apiModels, currentModel] = await Promise.all([
            getLocalModels(),
            getApiModels(),
            getCurrentModel()
        ]);

        // Vider le sélecteur
        select.innerHTML = "";

        // Ajouter les modèles locaux
        const localGroup = document.createElement("optgroup");
        localGroup.label = "Modèles locaux";

        localModels.forEach(model => {
            const option = document.createElement("option");
            option.value = model;
            option.textContent = model;
            localGroup.appendChild(option);
        });

        // Ajouter les modèles API
        const apiGroup = document.createElement("optgroup");
        apiGroup.label = "Modèles API";

        apiModels.forEach(model => {
            const option = document.createElement("option");
            option.value = model;
            option.textContent = model;
            apiGroup.appendChild(option);
        });

        select.appendChild(localGroup);
        select.appendChild(apiGroup);

        // Sélectionner le modèle actuellement utilisé
        select.value = currentModel.current_model;
        select.dataset.currentModel = currentModel.current_model;
        select.disabled = false;

        status.textContent = "Modèle actif";

    } catch (error) {
        status.textContent = "Impossible de charger les modèles.";
        console.error("Erreur de chargement des modèles :", error);
    }
}


// Réagir au changement de modèle
document.getElementById("modelSelect").addEventListener("change", async (event) => {
    const select = event.target;
    const status = document.getElementById("modelStatus");

    const selectedModel = select.value;
    const previousModel = select.dataset.currentModel;

    select.disabled = true;
    status.textContent = "Changement en cours...";

    try {
        const result = await changeModel(selectedModel);

        // Mémoriser le nouveau modèle actif
        select.dataset.currentModel = result.current_model;
        select.value = result.current_model;

        status.textContent = "Modèle changé avec succès.";

    } catch (error) {
        // Revenir au modèle précédent en cas d'échec
        select.value = previousModel;
        status.textContent = "Erreur lors du changement de modèle.";

        console.error(error);

    } finally {
        select.disabled = false;
    }
});

// Générer une réponse seule (sans évaluation)
async function generateAnswer(question, onChunk) {
    const params = new URLSearchParams({ question });

    const response = await fetch(
        `${API_URL}/generate?${params}`,
        { method: "POST" }
    );

    if (!response.ok) throw new Error("Erreur lors de la génération.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        onChunk(decoder.decode(value, { stream: true }));
    }
}

// Génération + évaluation avec streaming SSE
async function runFullPipeline(question, onGeneration, onEvaluation, onComplete) {
    const params = new URLSearchParams({ question });

    const response = await fetch(
        `${API_URL}/run-full-pipeline?${params}`,
        { method: "POST" }
    );

    if (!response.ok) throw new Error("Erreur lors de l'appel au pipeline.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let buffer = "";

    while (true) {
        const { value, done } = await reader.read();

        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

        const events = buffer.split("\n\n");
        buffer = events.pop();

        for (const event of events) {
            const dataLine = event
                .split("\n")
                .find(line => line.startsWith("data:"));

            if (!dataLine) continue;

            const data = JSON.parse(dataLine.slice(5).trim());

            if (data.error) {
                throw new Error(data.error);
            }

            if (data.type === "gen_chunk") {
                onGeneration?.(data.content);
            }

            if (data.type === "eval_chunk") {
                onEvaluation?.(data.content);
            }

            if (data.status === "completed") {
                onComplete?.(data);
            }
        }

        if (done) break;
    }
}



async function sendQuestionToAPI(question) {
    // Créer le bloc correspondant à cette conversation
    const turn = document.createElement("div");
    turn.className = "chat-turn";

    // Question de l'utilisateur
    const userMessage = document.createElement("div");
    userMessage.className = "user-message";
    userMessage.textContent = question;

    // Bloc de réponse de l'IA
    const answerSection = document.createElement("div");
    answerSection.className = "ai-response";

    const answerTitle = document.createElement("h3");
    answerTitle.textContent = "Réponse de l'IA";

    const answer = document.createElement("div");
    answer.className = "ai-response-content";
    answer.textContent = "Génération de la réponse...";

    answerSection.appendChild(answerTitle);
    answerSection.appendChild(answer);

    // Bloc du rapport d'évaluation
    const evaluationSection = document.createElement("div");
    evaluationSection.className = "ai-response";

    const evaluationTitle = document.createElement("h3");
    evaluationTitle.textContent = "Évaluation des biais";

    const evaluation = document.createElement("div");
    evaluation.className = "ai-response-content";
    evaluation.textContent = "En attente de l'évaluation...";

    evaluationSection.appendChild(evaluationTitle);
    evaluationSection.appendChild(evaluation);

    // Ajouter les blocs à la conversation
    turn.appendChild(userMessage);
    turn.appendChild(answerSection);
    turn.appendChild(evaluationSection);

    chatMessages.appendChild(turn);

    const chatContainer = document.querySelector(".chat-container");

    function scrollToBottom() {
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: "smooth"
        });
    }

    // Affiche le markdown en vrai HTML (tableaux, titres, gras).
    // Si les bibliothèques n'ont pas chargé (pas d'internet), on garde le texte brut.
    function renderMarkdown(element, text) {
        if (window.marked && window.DOMPurify) {
            element.innerHTML = DOMPurify.sanitize(marked.parse(text));
        } else {
            element.textContent = text;
        }
    }

    let answerText = "";
    let evaluationText = "";

    try {
        await runFullPipeline(
            question,

            // Réponse générée progressivement
            (chunk) => {
                answerText += chunk;
                renderMarkdown(answer, answerText);
                scrollToBottom();
            },

            // Rapport d'évaluation progressivement
            (chunk) => {
                evaluationText += chunk;
                renderMarkdown(evaluation, evaluationText);
                scrollToBottom();
            },

            // Fin du traitement
            (result) => {
                console.log("Pipeline terminé :", result);

                if (!evaluationText && result.evaluation_report) {
                    renderMarkdown(evaluation, result.evaluation_report);
                }

                scrollToBottom();
            }
        );

    } catch (error) {
        answer.textContent = "Erreur : " + error.message;
        evaluation.textContent = "L'évaluation n'a pas pu être effectuée.";

        console.error(error);
    }
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

    sendQuestionToAPI(question);

    input.value = "";
});


// Questions suivantes
document.getElementById("followupForm")
.addEventListener("submit", (e) => {

    e.preventDefault();

    const input = document.getElementById("followupInput");
    const question = input.value.trim();

    if (!question) return;

    sendQuestionToAPI(question);

    input.value = "";
});



initModelSelector();