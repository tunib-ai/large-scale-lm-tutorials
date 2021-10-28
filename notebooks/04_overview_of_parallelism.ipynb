{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Overview of Parallelism\n",
    "\n",
    "이번 세션에서는 병렬처리에 들어가기 앞서 다양한 병렬처리 기법을 전반적으로 살펴보겠습니다.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Parallelism\n",
    "병렬화란 여러개를 동시에 처리하는 기술을 의미하며 Large-scale 모델링에서 가장 중요한 기술 중 하나입니다. 머신러닝에서는 주로 여러개의 디바이스에서 연산을 병렬화 하여 속도나 메모리 효율성을 개선하기 위해 사용합니다.\n",
    "\n",
    "<br><br>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Parallelism\n",
    "데이터 병렬화는 데이터의 수가 많을 때, 데이터를 병렬처리하여 학습속도를 빠르게 하는 방법으로 모든 디바이스에 모델을 복제하고, 서로 다른 데이터를 각 디바이스에 입력하는 방식으로 동작합니다. 이로 인해 배치사이즈를 디바이스의 수의 배수만큼 더 많이 입력할 수 있습니다. 그러나 이러한 데이터 병렬화는 모델 하나가 디바이스 하나에 완전히 올라 갈 수 있을때 가능합니다.\n",
    "\n",
    "![](../images/data_parallelism.png)\n",
    "\n",
    "<br><br>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "## 3. Model Parallelism\n",
    "만약 모델이 너무 커서 하나의 디바이스에 완전히 올라갈 수 없다면, 파라미터를 쪼개서 여러 디바이스에 올려야 합니다. 따라서 각 디바이스에는 파라미터의 일부분들이 담겨있게 됩니다. 이로 인해 큰 모델도 작은 디바이스 여러개를 이용하면 처리가 가능해지며 병렬화 되는 차원에 따라 Inter-layer, Intra-layer 모델 병렬화로 구분할 수 있습니다.\n",
    "\n",
    "![](../images/model_parallelism.png)\n",
    "\n",
    "### Inter-layer Model Parallelism\n",
    "Inter-layer 모델 병렬화는 레이어를 기준으로 모델을 쪼개는 병렬화 방식입니다. 아래처럼 1,2,3번 레이어는 GPU 1번에, 4,5번 레이어는 GPU 2번에 할당 할 수 있으며, 대표적으로 Google의 GPipe가 이에 해당합니다.\n",
    "\n",
    "![](../images/inter_layer.png)\n",
    "\n",
    "### Intra-layer Model Parallelism\n",
    "인트라 레이어 모델 병렬화는 레이어와 상관 없이 텐서 자체를 쪼개는 병렬화 방식입니다. 예를 들면 [256, 256] 사이즈의 파라미터가 있다면 이를 [128, 256] 혹은 [256, 128]와 같이 쪼갤 수 있으며, 대표적으로 NVIDIA의 Megatron-LM이 이에 해당합니다.\n",
    "\n",
    "![](../images/intra_layer.png)\n",
    "\n",
    "### Pipeline Parallelism\n",
    "파이프라인 병렬화는 인터 레이어 모델 병렬화의 단점을 개선한 모델 병렬화 기법입니다. 인터 레이어 모델 병렬화를 수행할때 반드시 GPU의 연산 순서가 생깁니다. 예를 들어 1,2,3 레이어가 실행되지 못한다면 4,5레이어는 실행될 수 없으므로 GPU 1번의 연산이 끝날때까지 GPU 2번은 기다려야 합니다.\n",
    "\n",
    "![](../images/pipeline_parallelism.png)\n",
    "\n",
    "<br>\n",
    "\n",
    "이는 매우 비효율적입니다. GPU가 여러대 있지만 실제로는 동시에 하나의 GPU만 활용할 수 있기 때문이죠. 이러한 문제를 해결하기 위해 아래처럼 연산과정을 병렬적으로 파이프라이닝하는 것이 파이프라인 병렬화입니다. (말이 어렵죠? 뒤에서 자세히 알려드릴게요.)\n",
    "\n",
    "<br>\n",
    "\n",
    "![](../images/pipeline_parallelism2.png)\n",
    "\n",
    "<br><br>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Multi-dimensional Parallelism\n",
    "위에서 언급한 다양한 병렬화 기법들은 동시에 여러개를 적용할 수도 있으며 적용되는 병렬화의 개수에 따라 차원이 늘어납니다. 아래와 같이 다양한 방법으로 n-차원 병렬화를 수행 할 수 있죠.\n",
    "\n",
    "- e.g. 2차원 병렬화: 데이터 병렬화 + 인터 레이어 병렬화\n",
    "- e.g. 2차원 병렬화: 데이터 병렬화 + 인트라 레이어 병렬화\n",
    "- e.g. 3차원 병렬화: 데이터 병렬화 + 인트라 레이어 병렬화 + 파이프라인 병렬화\n",
    "\n",
    "![](../images/parallelism.png)\n",
    "\n",
    "이러한 다차원 병렬화는 요즘 large-scale 모델링에서 가장 각광받고 있는 방식입니다. 위에서 언급한 방법들 이외에 ZeRO 등의 기법이 추가로 존재합니다만 이는 나중 챕터에서 자세히 설명하겠습니다."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
