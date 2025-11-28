import sys
sys.path.append('/home/bsh/Woorifisa5_final/mcp-server')

from Cluster.clustering import predict_cluster

# 테스트 데이터 (df_clustering_result.csv의 첫 번째 행 - 클러스터 4)
test_data = {
    '이용건수_신판_R3M': 5,
    '이용금액_신판_R3M': 382923,
    '이용금액_쇼핑': 0,
    '이용금액_요식': 0,
    '이용금액_교통': 0,
    '이용금액_의료': 0,
    '이용금액_납부': 197294,
    '이용금액_교육': 0,
    '이용금액_여유생활': 0,
    '이용금액_사교활동': 0,
    '이용금액_일상생활': 0,
    '이용금액_해외': 0
}

print("테스트 시작...")
print(f"입력 데이터: {test_data}")
print()

cluster_id = predict_cluster(test_data)
print(f"\n최종 결과: 클러스터 {cluster_id}")
print("테스트 완료!")
