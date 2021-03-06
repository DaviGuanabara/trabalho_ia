# Trabalho de IA

## Descrição

Esse é o trabalho está sendo desenvolvido para as cadeiras de IA do Mestrado Profissional.

Escopo: Aplicar RL em Mario Bros
Data de entrega: 30/06/2022
Grupo:

1. Andrey Morais Labanca
2. Davi Guanabara de Aragão
3. Tiago Aroeira Marlieri

Relatório: https://onedrive.live.com/edit.aspx?resid=204284DD53F1A1E2!2129461&ithint=file%2cdocx&authkey=!AKDXhfOaPectlaM \\
Apresentação: https://docs.google.com/presentation/d/17BRWOBCa9FX8glRyxFHZnFeN3B5pfAc9ExwirHoWh5w/edit?usp=sharing \\

Modelos Treinados: https://drive.google.com/drive/folders/1-Sa4U-pps3cJLzbkUTkqVJZ1RtqDV6PK?usp=sharing

## Conteúdos Relacionados

#Mais ambientes
https://github.com/clvrai/awesome-rl-envs

Dicas Stable Baselines 3
https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

Stable Baselines 3 Zoo:
https://github.com/DLR-RM/rl-baselines3-zoo

Stable Baselines 3 GitHub:
https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

Guia para aplicação de RL:
https://towardsdatascience.com/a-beginners-guide-to-reinforcement-learning-with-a-mario-bros-example-fa0e0563aeb7

Guia indicado pelo professor Marcos:
https://pythonprogramming.net/introduction-reinforcement-learning-stable-baselines-3-tutorial/

Site de possíveis jogos:
https://www.gymlibrary.ml/#

Blog sobre aplicação de RL em super mário:
https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/

Vídeo da aplicação:
https://www.youtube.com/watch?v=qv6UVOQ0F44


## Instalação do Ambiente

No mac.

1. Instalar PIP
2. (Não é necessário se for só Atari) Instalar SWIG - brew install swig (se não, dá problema com o GYM https://github.com/openai/spinningup/issues/32)
3. Instalar GYM Atari - pip install gym[atari] (https://github.com/openai/gym) (para instalar ALL, precisa de licensa pro mujoco, https://amulyareddyk97.medium.com/mujoco-setup-on-macos-667ca5efee68, além do que dá moh problema de instalar ele).
4. Instalar Baselines3 - pip install stable-baselines3[extra] (https://stable-baselines3.readthedocs.io/en/master/guide/install.html)
5. Instalar Super Mario Bros - pip install gym-super-mario-bros (https://pypi.org/project/gym-super-mario-bros/)

## Lista de melhorias:
https://github.com/DaviGuanabara/trabalho_ia/tree/main/supermario

## Dicas

Rodar o tensorboard:
$ tensorboard --logdir=logs
Ele vai gerar um link para ser visualizado pelo navegador

## Histórico de decisões
### Dia 25/04/2022

Decidimos sobre a divisão do trabalho e o toy problem:

#### Toy Problem: 
Mario Bros NES

#### Divisão do Trabalho

Tiago: API

Falta distribuir entre o Davi e o Andrey as seguintes RL: PPO e SAC

### Dia 06/06/2022

Divisão de trabalho Atualizada!
1. Andrey: IA RL PPO
2. Davi: IA RL SAC
3. Tiago: API do jogo 

### Dia 10/06/2022

Escolha do Simulador:
Gym Retro
https://openai.com/blog/gym-retro/


### Dia 22/06/2022

Não conseguimos adaptar o SAC para um conjunto de ações discretas (necessárias para o super mario). Assim, vamos substituir o SAC para o DQN.

