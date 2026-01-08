package fr.mael3il.api;

import fr.mael3il.objets.Robot;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.Response.ResponseBuilder;
import org.eclipse.microprofile.context.ThreadContext;
import org.neo4j.driver.Driver;
import org.neo4j.driver.Values;
import org.neo4j.driver.async.AsyncSession;

import java.util.List;
import java.util.concurrent.CompletionStage;

@Path("api/v1/robots")
@Consumes(MediaType.APPLICATION_JSON)
@Produces(MediaType.APPLICATION_JSON)
public class ApiRobots {

    @Inject
    Driver driver;

    @Inject
    ThreadContext threadContext;

    @GET
    public CompletionStage<Response> getAllRobots() {
        AsyncSession session = driver.session(AsyncSession.class);

        CompletionStage<List<Robot>> cs = session
                .executeReadAsync(tx -> tx
                        .runAsync("MATCH (r:Robot) RETURN r")
                        .thenCompose(cursor ->
                                cursor.listAsync(record ->
                                        Robot.from(record.get("r").asNode())
                                )));

        return threadContext.withContextCapture(cs)
                .thenCompose(robots ->
                        session.closeAsync().thenApply(signal -> robots))
                .thenApply(Response::ok)
                .thenApply(ResponseBuilder::build);
    }

    @GET
    @Path("/{id}")
    public CompletionStage<Response> getRobotById(@PathParam("id") String id) {
        AsyncSession session = driver.session(AsyncSession.class);

        CompletionStage<Robot> cs = session
                .executeReadAsync(tx -> tx
                        .runAsync(
                                "MATCH (r:Robot {id: $id}) RETURN r",
                                Values.parameters("id", id)
                        )
                        .thenCompose(cursor ->
                                cursor.singleAsync()
                                        .thenApply(record ->
                                                Robot.from(record.get("r").asNode())
                                        )));

        return threadContext.withContextCapture(cs)
                .thenCompose(robot ->
                        session.closeAsync().thenApply(signal -> robot))
                .thenApply(robot -> {
                    if (robot == null) {
                        return Response.status(Response.Status.NOT_FOUND).build();
                    }
                    return Response.ok(robot).build();
                })
                .exceptionally(ex ->
                        Response.status(Response.Status.NOT_FOUND).build());
    }

    @POST
    public CompletionStage<Response> createRobot(Robot robot) {
        AsyncSession session = driver.session(AsyncSession.class);

        CompletionStage<Robot> cs = session
                .executeWriteAsync(tx -> tx
                        .runAsync(
                                """
                                CREATE (r:Robot:AgriculturalDevice:SAREF_Device:Sensor {
                                  id: randomUUID(),
                                  modele: $modele,
                                  type: $type,
                                  weed_detection_threshold: $threshold,
                                  statut: $statut
                                })
                                RETURN r
                                """,
                                Values.parameters(
                                        "modele", robot.getModele(),
                                        "type", robot.getType(),
                                        "threshold", robot.getWeedDetectionThreshold(),
                                        "statut", robot.getStatut()
                                ))
                        .thenCompose(cursor ->
                                cursor.singleAsync()
                                        .thenApply(record ->
                                                Robot.from(record.get("r").asNode())
                                        )));

        return threadContext.withContextCapture(cs)
                .thenCompose(created ->
                        session.closeAsync().thenApply(signal -> created))
                .thenApply(created ->
                        Response.status(Response.Status.CREATED)
                                .entity(created)
                                .build());
    }

    @PUT
    @Path("/{id}")
    public CompletionStage<Response> updateRobot(@PathParam("id") String id, Robot robot) {
        AsyncSession session = driver.session(AsyncSession.class);

        CompletionStage<Robot> cs = session
                .executeWriteAsync(tx -> tx
                        .runAsync(
                                """
                                MATCH (r:Robot {id: $id})
                                SET r.modele = $modele,
                                    r.type = $type,
                                    r.weed_detection_threshold = $threshold,
                                    r.statut = $statut
                                RETURN r
                                """,
                                Values.parameters(
                                        "id", id,
                                        "modele", robot.getModele(),
                                        "type", robot.getType(),
                                        "threshold", robot.getWeedDetectionThreshold(),
                                        "statut", robot.getStatut()
                                ))
                        .thenCompose(cursor ->
                                cursor.singleAsync()
                                        .thenApply(record ->
                                                Robot.from(record.get("r").asNode())
                                        )));

        return threadContext.withContextCapture(cs)
                .thenCompose(updated ->
                        session.closeAsync().thenApply(signal -> updated))
                .thenApply(updated ->
                        Response.ok(updated).build());
    }

    @DELETE
    @Path("/{id}")
    public CompletionStage<Response> deleteRobot(@PathParam("id") String id) {
        AsyncSession session = driver.session(AsyncSession.class);

        CompletionStage<Void> cs = session
                .executeWriteAsync(tx -> tx
                        .runAsync(
                                "MATCH (r:Robot {id: $id}) DELETE r",
                                Values.parameters("id", id)
                        )
                        .thenApply(r -> null));

        return threadContext.withContextCapture(cs)
                .thenCompose(signal ->
                        session.closeAsync().thenApply(s -> signal))
                .thenApply(signal ->
                        Response.ok().build());
    }
}
