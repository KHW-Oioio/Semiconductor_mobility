import streamlit as st
st.title('방명록!')
name = st.text_input('이름을 입력하세요 : ')
menu = st.selectbox('신분 :', ['학생', '선생님'])
if st.button('인사말 생성'):
  st.write(name + '님! 당신은 ' + menu + '이시군요 반갑습니다')
