import streamlit as st
from groq import Groq

# Inicializar o cliente Groq com a chave da API
client = Groq(
    api_key="GROQ_API_KEY3")


def get_response_from_model(message):
    try:
        # Chamada para o modelo
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "user", "content": message}
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        response_content = ""
        for chunk in completion:
            response_content += chunk.choices[0].delta.content or ""

        return response_content.strip()
    except Exception as e:
        return f"Erro ao obter resposta: {e}"


def main():
    st.set_page_config(layout='wide')

    st.title("Chatbot Test")

    # Histórico de mensagens
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibir histórico de mensagens
    for user_message, bot_response in st.session_state.messages:
        with st.chat_message("user"):
            st.write(user_message)
        with st.chat_message("assistant"):
            st.write(bot_response)

    # Entrada do usuário
    new_message = st.chat_input("Digite uma mensagem")
    if new_message:
        # Adicionar mensagem do usuário ao histórico
        st.session_state.messages.append((new_message, ""))
        # Obter resposta do modelo
        response = get_response_from_model(new_message)
        # Atualizar histórico com a resposta do modelo
        st.session_state.messages[-1] = (new_message, response)
        # Mostrar resposta
        st.chat_message("user").write(new_message)
        st.chat_message("assistant").write(response)


if __name__ == "__main__":
    main()
